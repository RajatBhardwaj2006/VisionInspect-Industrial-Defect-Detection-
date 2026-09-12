# File path: X:\VScode\Artificial_intelligence_n_Machine_learning\VisionInspect\src\detection\anomaly_detector.py

import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T

from src.detection.anomaly_score import reconstruction_loss
from src.detection.heatmap import compute_anomaly_map
from src.detection.localization import localize
from src.utils.config import load_config

class AnomalyDetector:
    def __init__(self, category='bottle', model_path=None, device=None):
        self.category = category
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_path = Path(model_path) if model_path else Path(f"models/{category}/autoencoder.pth")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")
            
        from src.models.autoencoder import Autoencoder
        self.model = Autoencoder()
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load configs
        self.config = load_config()
        self.image_threshold = self.config["detection"]["image_threshold"]
        self.pixel_threshold = self.config["detection"]["pixel_threshold"]
        self.image_score_method = self.config["detection"].get("image_score_method", "max_raw")

    def inspect(self, image_path: str):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
            
        original_pil = Image.open(image_path).convert('RGB')
        original_np = np.array(original_pil)
        orig_h, orig_w, _ = original_np.shape

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        
        img_tensor = transform(original_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed = self.model(img_tensor)
            amap = compute_anomaly_map(img_tensor, reconstructed).squeeze(0)
            score = float(amap.max().item())
            
        # Run localization to extract all valid regions
        regions = localize(amap, pixel_thresh=self.pixel_threshold)
        
        # Classification rule based on model-derived max raw score and localized regions check
        is_defective = (score > self.image_threshold) and (len(regions) > 0)
        
        # Safeguard region mass check to filter out reconstruction noise
        if is_defective:
            top_reg = regions[0]
            if top_reg["total_mass"] < 25.0 and top_reg["score"] < 0.17:
                is_defective = False
                
        status = "DEFECTIVE" if is_defective else "NORMAL"

        # Scale regions back to original image dimensions
        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0
        
        scaled_regions = []
        for reg in regions:
            scaled_regions.append({
                "x": int(reg["x"] * scale_x),
                "y": int(reg["y"] * scale_y),
                "width": int(reg["width"] * scale_x),
                "height": int(reg["height"] * scale_y),
                "area": int(reg["area"] * scale_x * scale_y),
                "centroid": (float(reg["centroid"][0] * scale_x), float(reg["centroid"][1] * scale_y)),
                "score": reg.get("score", 0.0),
                "max_val": reg.get("max_val", 0.0),
                "total_mass": reg.get("total_mass", 0.0)
            })

        bounding_box = None
        if is_defective and scaled_regions:
            bounding_box = scaled_regions[0]

        defect_type = image_path.parent.name
        results_dir = Path("results/predictions")
        results_dir.mkdir(parents=True, exist_ok=True)
        saved_path = results_dir / f"{self.category}_{defect_type}_{image_path.stem}_result.png"

        amap_np = amap.detach().cpu().numpy()
        amap_resized = cv2.resize(amap_np, (orig_w, orig_h))
        heatmap_img = plt.get_cmap('jet')(amap_resized)[:, :, :3]
        overlay = (0.5 * original_np / 255.0 + 0.5 * heatmap_img)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(overlay, 0, 1))
        if is_defective:
            for r in scaled_regions:
                bx, by, bw, bh = r["x"], r["y"], r["width"], r["height"]
                rect = plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)
        plt.axis('off')
        plt.savefig(saved_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        explanation = (
            f"Anomaly score {score:.4f} exceeds threshold {self.image_threshold:.4f} and localized regions were found."
            if is_defective else f"Anomaly score {score:.4f} is normal or no valid regions were localized."
        )

        return {
            "status": status,
            "score": score,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "bounding_box": bounding_box or "None",
            "localized_regions": scaled_regions if is_defective else [],
            "explanation": explanation,
            "saved_path": str(saved_path)
        }