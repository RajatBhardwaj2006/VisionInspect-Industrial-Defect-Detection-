import os
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
from src.detection.patchcore_v22_detector import PatchCoreDetectorV22
from src.utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    category = "bottle"
    dataset_root = Path("dataset/mvtec_anomaly_detection")
    bottle_test = dataset_root / category / "test"
    bottle_gt = dataset_root / category / "ground_truth"
    
    detector_v23 = PatchCoreDetectorV23(category=category, device=device)
    detector_v22 = PatchCoreDetectorV22(category=category, device=device)
    
    out_visuals = Path("results/phase2/phase2_3/visuals")
    out_comparisons = Path("results/phase2/phase2_3/comparisons")
    out_visuals.mkdir(parents=True, exist_ok=True)
    out_comparisons.mkdir(parents=True, exist_ok=True)
    
    target_images = [
        ("good", "001.png"),
        ("good", "002.png"),
        ("broken_large", "000.png"),
        ("broken_large", "004.png"),
        ("broken_small", "000.png"),
        ("broken_small", "003.png"),
        ("contamination", "000.png"),
        ("contamination", "002.png"),
    ]
    
    for defect_type, img_name in target_images:
        img_path = bottle_test / defect_type / img_name
        if not img_path.exists():
            print(f"File not found: {img_path}")
            continue
            
        gt_path = bottle_gt / defect_type / img_name.replace(".png", "_mask.png")
        has_gt = gt_path.exists()
        stem = img_name.split('.')[0]
        
        # Run inspection
        res_v23 = detector_v23.inspect(str(img_path))
        res_v22 = detector_v22.inspect(str(img_path))
        
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        orig_h, orig_w, _ = img_np.shape
        
        # 6-panel visualization:
        # Original | Anomaly Heatmap | Threshold Mask | Final Localization | Ground Truth Mask | Pred vs GT Compare
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Panel 1: Original
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(f"Original ({defect_type}/{img_name})")
        axes[0, 0].axis("off")
        
        # Panel 2: Heatmap
        v23_img = cv2.imread(res_v23['saved_path'])
        if v23_img is not None:
            axes[0, 1].imshow(cv2.cvtColor(v23_img, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 1].imshow(img_np)
        axes[0, 1].set_title(f"Phase 2.3 Anomaly Heatmap\nScore: {res_v23['score']:.4f}")
        axes[0, 1].axis("off")
        
        # Panel 3: Threshold Mask (Clean Mask)
        clean_mask = res_v23.get("clean_mask", np.zeros((256, 256), dtype=bool))
        axes[0, 2].imshow(clean_mask, cmap="gray")
        axes[0, 2].set_title("Threshold Mask (Pixel Mask)")
        axes[0, 2].axis("off")
        
        # Panel 4: Ground Truth Mask
        if has_gt:
            gt_pil = Image.open(gt_path).convert("L")
            gt_np = cv2.resize(np.array(gt_pil), (orig_w, orig_h))
            axes[1, 0].imshow(gt_np, cmap="gray")
            axes[1, 0].set_title("Ground Truth Mask")
        else:
            axes[1, 0].imshow(np.zeros((orig_h, orig_w)), cmap="gray")
            axes[1, 0].set_title("Ground Truth (Normal)")
        axes[1, 0].axis("off")
        
        # Panel 5: Phase 2.2 Baseline Overlay
        v22_img = cv2.imread(res_v22['saved_path'])
        if v22_img is not None:
            axes[1, 1].imshow(cv2.cvtColor(v22_img, cv2.COLOR_BGR2RGB))
        else:
            axes[1, 1].imshow(img_np)
        axes[1, 1].set_title(f"Phase 2.2 Baseline Overlay\n[{res_v22['status']}]")
        axes[1, 1].axis("off")
        
        # Panel 6: Pred vs GT Compare (Green = GT, Red = Pred)
        compare_img = img_np.copy()
        if has_gt:
            gt_bin = (gt_np > 0)
            compare_img[gt_bin] = [0, 255, 0] # Green GT
        pred_bin = cv2.resize(clean_mask.astype(np.uint8), (orig_w, orig_h)) > 0
        if res_v23['status'] == "DEFECTIVE":
            compare_img[pred_bin] = [255, 0, 0] # Red Pred
            overlap = np.logical_and(gt_bin if has_gt else False, pred_bin)
            compare_img[overlap] = [255, 255, 0] # Yellow Overlap
            
        axes[1, 2].imshow(compare_img)
        axes[1, 2].set_title(f"Pred (Red) vs GT (Green)\nOverlap (Yellow)")
        axes[1, 2].axis("off")
        
        plt.tight_layout()
        save_path = out_visuals / f"bottle_{defect_type}_{stem}_v23_panel.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        if v23_img is not None:
            cv2.imwrite(str(out_comparisons / f"bottle_{defect_type}_{stem}_phase2_3.png"), v23_img)
            
        print(f"Saved Phase 2.3 visual comparison for {defect_type}/{img_name} -> Status: {res_v23['status']} (Regions: {len(res_v23['localized_regions'])})")

if __name__ == "__main__":
    main()
