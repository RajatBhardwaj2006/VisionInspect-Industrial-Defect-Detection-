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

from src.detection.patchcore_v22_detector import PatchCoreDetectorV22
from src.detection.patchcore_detector import PatchCoreDetector
from src.detection.anomaly_detector import AnomalyDetector
from src.utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    category = "bottle"
    dataset_root = Path("dataset/mvtec_anomaly_detection")
    bottle_test = dataset_root / category / "test"
    bottle_gt = dataset_root / category / "ground_truth"
    
    detector_v22 = PatchCoreDetectorV22(category=category, device=device)
    detector_p2 = PatchCoreDetector(category=category, device=device)
    detector_p1 = AnomalyDetector(category=category, device=device)
    
    out_visuals = Path("results/phase2/phase2_2_visuals")
    out_comparisons = Path("results/phase2/phase2_2_comparisons")
    out_visuals.mkdir(parents=True, exist_ok=True)
    out_comparisons.mkdir(parents=True, exist_ok=True)
    
    # 3 good, 3 broken_large, 3 broken_small, 3 contamination
    target_images = [
        ("good", "001.png"),
        ("good", "002.png"),
        ("good", "003.png"),
        ("broken_large", "000.png"),
        ("broken_large", "004.png"),
        ("broken_large", "005.png"),
        ("broken_small", "000.png"),
        ("broken_small", "003.png"),
        ("broken_small", "005.png"),
        ("contamination", "000.png"),
        ("contamination", "002.png"),
        ("contamination", "004.png")
    ]
    
    for defect_type, img_name in target_images:
        img_path = bottle_test / defect_type / img_name
        if not img_path.exists():
            print(f"File not found: {img_path}")
            continue
            
        gt_path = bottle_gt / defect_type / img_name.replace(".png", "_mask.png")
        has_gt = gt_path.exists()
        stem = img_name.split('.')[0]
        
        # Run inspection for all 3 detectors
        res_v22 = detector_v22.inspect(str(img_path))
        res_p2 = detector_p2.inspect(str(img_path))
        res_p1 = detector_p1.inspect(str(img_path))
        
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        orig_h, orig_w, _ = img_np.shape
        
        # 1. Generate detailed 5-panel visualization for V2.2
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # Panel 1: Original
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original ({defect_type}/{img_name})")
        axes[0].axis("off")
        
        # Panel 2: Ground Truth Mask
        if has_gt:
            gt_pil = Image.open(gt_path).convert("L")
            gt_np = cv2.resize(np.array(gt_pil), (orig_w, orig_h))
            axes[1].imshow(gt_np, cmap="gray")
            axes[1].set_title("Ground Truth Mask")
        else:
            axes[1].imshow(np.zeros((orig_h, orig_w)), cmap="gray")
            axes[1].set_title("Ground Truth (Normal)")
        axes[1].axis("off")
        
        # Panel 3: Phase 1 Autoencoder Overlay
        p1_img = cv2.imread(res_p1['saved_path'])
        if p1_img is not None:
            axes[2].imshow(cv2.cvtColor(p1_img, cv2.COLOR_BGR2RGB))
        else:
            axes[2].imshow(img_np)
        axes[2].set_title(f"Phase 1 Autoencoder\n[{res_p1['status']}]")
        axes[2].axis("off")
        
        # Panel 4: Phase 2 Baseline PatchCore Overlay
        p2_img = cv2.imread(res_p2['saved_path'])
        if p2_img is not None:
            axes[3].imshow(cv2.cvtColor(p2_img, cv2.COLOR_BGR2RGB))
        else:
            axes[3].imshow(img_np)
        axes[3].set_title(f"Phase 2 PatchCore Baseline\n[{res_p2['status']}]")
        axes[3].axis("off")
        
        # Panel 5: Phase 2.2 Crack-Sensitive PatchCore Overlay
        v22_img = cv2.imread(res_v22['saved_path'])
        if v22_img is not None:
            axes[4].imshow(cv2.cvtColor(v22_img, cv2.COLOR_BGR2RGB))
        else:
            axes[4].imshow(img_np)
        axes[4].set_title(f"Phase 2.2 Crack-Sensitive\n[{res_v22['status']}] Score: {res_v22['score']:.2f}")
        axes[4].axis("off")
        
        plt.tight_layout()
        save_path = out_visuals / f"bottle_{defect_type}_{stem}_v22_full.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        
        # Also copy clean overlay to comparisons folder
        if v22_img is not None:
            cv2.imwrite(str(out_comparisons / f"bottle_{defect_type}_{stem}_phase2_2.png"), v22_img)
            
        print(f"Saved visual comparison for {defect_type}/{img_name} -> V2.2 Status: {res_v22['status']} (Regions: {len(res_v22['localized_regions'])})")

if __name__ == "__main__":
    main()
