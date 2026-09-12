import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

from src.models.patchcore import PatchCoreModel
from src.detection.localization import localize, postprocess_anomaly_map
from src.utils.config import load_config

class PatchCoreDetector:
    """
    Phase 2 PatchCore Feature-Based Anomaly Detector.
    Wraps PatchCoreModel for image inspection and multi-region localization.
    """
    def __init__(self, category: str = 'bottle', model_dir: Optional[str] = None, device: Optional[torch.device] = None):
        self.category = category
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        self.config = load_config()
        pc_cfg = self.config.get("patchcore", {})
        
        dir_path = Path(model_dir) if model_dir else Path(pc_cfg.get("model_dir", f"models/{category}/patchcore"))
        if not dir_path.exists():
            raise FileNotFoundError(f"PatchCore model directory not found at: {dir_path}")
            
        self.model = PatchCoreModel(device=self.device)
        self.model.load(str(dir_path))
        
        self.image_threshold = pc_cfg.get("image_threshold", 0.35)
        self.pixel_threshold = pc_cfg.get("pixel_threshold", 0.25)
        self.gaussian_sigma = pc_cfg.get("gaussian_sigma", 4.0)

    def inspect(self, image_path: str) -> Dict[str, Any]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found at: {image_path}")
            
        original_pil = Image.open(image_path).convert('RGB')
        original_np = np.array(original_pil)
        orig_h, orig_w, _ = original_np.shape

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        
        img_tensor = transform(original_pil).unsqueeze(0).to(self.device)

        # Predict patchcore score and anomaly map
        raw_score, amap_tensor = self.model.predict(img_tensor)
        
        # Apply Gaussian smoothing to PatchCore anomaly map
        amap_smoothed = postprocess_anomaly_map(amap_tensor, sigma=self.gaussian_sigma)
        
        # Run localization using Phase 1 localization engine
        regions = localize(amap_smoothed, pixel_thresh=self.pixel_threshold)
        
        score = float(amap_smoothed.max().item())
        is_defective = (score > self.image_threshold) and (len(regions) > 0)
        
        # Region mass check safeguard
        if is_defective and regions:
            top_reg = regions[0]
            if top_reg["total_mass"] < 15.0 and top_reg["score"] < 0.15:
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

        bounding_box = scaled_regions[0] if (is_defective and scaled_regions) else None

        defect_type = image_path.parent.name
        results_dir = Path("results/predictions")
        results_dir.mkdir(parents=True, exist_ok=True)
        saved_path = results_dir / f"patchcore_{self.category}_{defect_type}_{image_path.stem}_result.png"

        amap_np = amap_smoothed.detach().cpu().numpy()
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
            f"PatchCore feature distance {score:.4f} exceeds threshold {self.image_threshold:.4f} and localized regions were found."
            if is_defective else f"PatchCore anomaly score {score:.4f} is normal or no valid regions were localized."
        )

        return {
            "category": self.category,
            "status": status,
            "score": score,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "bounding_box": bounding_box or "None",
            "localized_regions": scaled_regions if is_defective else [],
            "explanation": explanation,
            "saved_path": str(saved_path)
        }
