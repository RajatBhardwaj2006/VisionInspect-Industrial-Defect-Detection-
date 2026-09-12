import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.autoencoder import Autoencoder
from src.detection.heatmap import compute_anomaly_map
from src.detection.localization import localize
from src.utils.config import load_config
from torchvision import transforms

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = Path("models/bottle/autoencoder.pth")
    model = Autoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    image_size = cfg.get("model", {}).get("image_size", 256)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    pixel_threshold = cfg["detection"]["pixel_threshold"]
    image_threshold = cfg["detection"]["image_threshold"]
    
    out_dir = Path("results/phase1_final/visuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_root = Path("dataset/mvtec_anomaly_detection")
    bottle_test = dataset_root / "bottle" / "test"
    bottle_gt = dataset_root / "bottle" / "ground_truth"
    
    # 3 normal, 5 defective (including the required ones)
    target_images = [
        ("good", "001.png"),
        ("good", "002.png"),
        ("good", "003.png"),
        ("broken_large", "000.png"),
        ("broken_small", "000.png"),
        ("contamination", "000.png"),
        ("broken_large", "004.png"),
        ("broken_small", "005.png"),
    ]
    
    for category, img_name in target_images:
        img_path = bottle_test / category / img_name
        if not img_path.exists():
            print(f"File not found: {img_path}")
            continue
            
        gt_path = bottle_gt / category / img_name.replace(".png", "_mask.png")
        has_gt = gt_path.exists()
        
        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        orig_img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Preprocess
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon = model(img_tensor)
            amap_tensor = compute_anomaly_map(img_tensor, recon).squeeze(0)
            amap = amap_tensor.cpu().numpy()
            
            recon_img = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_img = (np.clip(recon_img, 0, 1) * 255).astype(np.uint8)
            
        img_score = float(amap.max())
        
        regions = localize(amap_tensor, pixel_thresh=pixel_threshold)
        
        is_def = (img_score > image_threshold) and (len(regions) > 0)
        if is_def:
            top_reg = regions[0]
            if top_reg["total_mass"] < 25.0 and top_reg["score"] < 0.17:
                is_def = False
                
        # Prepare binary mask
        pred_mask = np.zeros((256, 256), dtype=np.uint8)
        if is_def and regions:
            for r in regions:
                x, y, w, h = r["x"], r["y"], r["width"], r["height"]
                pred_mask[y:y+h, x:x+w] = 255
                
        # Visualization 1: Combine them into one plot
        fig, axes = plt.subplots(1, 6 if has_gt else 5, figsize=(20, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        ax_idx = 0
        
        # Original
        axes[ax_idx].imshow(img_pil)
        axes[ax_idx].set_title("Original")
        axes[ax_idx].axis("off")
        ax_idx += 1
        
        # Reconstruction
        axes[ax_idx].imshow(recon_img)
        axes[ax_idx].set_title("Reconstruction")
        axes[ax_idx].axis("off")
        ax_idx += 1
        
        # Raw Heatmap
        im = axes[ax_idx].imshow(amap, cmap="jet")
        axes[ax_idx].set_title("Raw Heatmap")
        axes[ax_idx].axis("off")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        ax_idx += 1
        
        # Processed Mask
        axes[ax_idx].imshow(pred_mask, cmap="gray")
        axes[ax_idx].set_title("Predicted Mask")
        axes[ax_idx].axis("off")
        ax_idx += 1
        
        # Ground Truth
        if has_gt:
            gt_pil = Image.open(gt_path).convert("L")
            gt_np = np.array(gt_pil)
            gt_np = cv2.resize(gt_np, (256, 256))
            axes[ax_idx].imshow(gt_np, cmap="gray")
            axes[ax_idx].set_title("Ground Truth")
            axes[ax_idx].axis("off")
            ax_idx += 1
            
        # Highlighted Final
        final_highlight = orig_img_cv2.copy()
        orig_h, orig_w = final_highlight.shape[:2]
        
        if is_def and regions:
            for r in regions:
                # Scale coordinates to original size
                scale_x = orig_w / 256
                scale_y = orig_h / 256
                x = int(r["x"] * scale_x)
                y = int(r["y"] * scale_y)
                w = int(r["width"] * scale_x)
                h = int(r["height"] * scale_y)
                
                cv2.rectangle(final_highlight, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
        axes[ax_idx].imshow(cv2.cvtColor(final_highlight, cv2.COLOR_BGR2RGB))
        axes[ax_idx].set_title("Final Highlighted")
        axes[ax_idx].axis("off")
        
        plt.tight_layout()
        stem = img_name.split('.')[0]
        out_name = f"bottle_{category}_{stem}_result.png"
        plt.savefig(out_dir / out_name, dpi=150)
        plt.close(fig)
        
        print(f"Status: {'DEFECTIVE' if is_def else 'NORMAL'}")
        print(f"Anomaly Score: {img_score:.6f}")
        print(f"Image Threshold: {image_threshold:.6f}")
        print(f"Pixel Threshold: {pixel_threshold:.6f}")
        print(f"Number of Regions: {len(regions)}")
        for r in regions:
            print(f"  Box: x={r['x']}, y={r['y']}, w={r['width']}, h={r['height']}, mass={r['total_mass']:.2f}")
        print(f"Saved: {out_dir / out_name}")
        print("-" * 50)

if __name__ == '__main__':
    main()
