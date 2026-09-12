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

from src.detection.anomaly_detector import AnomalyDetector
from src.detection.patchcore_detector import PatchCoreDetector
from src.utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    category = "bottle"
    dataset_root = Path("dataset/mvtec_anomaly_detection")
    bottle_test = dataset_root / category / "test"
    
    print("Initializing detectors...")
    ae_detector = AnomalyDetector(category=category, device=device)
    pc_detector = PatchCoreDetector(category=category, device=device)
    
    out_dir = Path("results/phase2/comparisons")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_images = [
        ("good", "001.png"),
        ("good", "002.png"),
        ("broken_large", "000.png"),
        ("broken_small", "000.png"),
        ("contamination", "000.png"),
        ("broken_large", "004.png"),
        ("contamination", "002.png"),
    ]
    
    for defect_type, img_name in target_images:
        img_path = bottle_test / defect_type / img_name
        if not img_path.exists():
            print(f"File not found: {img_path}")
            continue
            
        stem = img_name.split('.')[0]
        
        # Run Phase 1 Autoencoder inspection
        ae_res = ae_detector.inspect(str(img_path))
        
        # Run Phase 2 PatchCore inspection
        pc_res = pc_detector.inspect(str(img_path))
        
        print(f"\n==================== Image: {defect_type}/{img_name} ====================")
        print(f"Phase 1 Autoencoder -> Status: {ae_res['status']}, Score: {ae_res['score']:.4f}, Regions: {len(ae_res['localized_regions'])}")
        print(f"Phase 2 PatchCore   -> Status: {pc_res['status']}, Score: {pc_res['score']:.4f}, Regions: {len(pc_res['localized_regions'])}")
        
        # Save side-by-side plot comparing Phase 1 and Phase 2
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original Image ({defect_type}/{img_name})")
        axes[0].axis("off")
        
        # 2. Phase 1 Autoencoder Output
        ae_img = cv2.imread(ae_res['saved_path'])
        if ae_img is not None:
            ae_img_rgb = cv2.cvtColor(ae_img, cv2.COLOR_BGR2RGB)
            axes[1].imshow(ae_img_rgb)
        else:
            axes[1].imshow(img_np)
        axes[1].set_title(f"Phase 1 Autoencoder\n[{ae_res['status']}] Score: {ae_res['score']:.4f}")
        axes[1].axis("off")
        
        # 3. Phase 2 PatchCore Output
        pc_img = cv2.imread(pc_res['saved_path'])
        if pc_img is not None:
            pc_img_rgb = cv2.cvtColor(pc_img, cv2.COLOR_BGR2RGB)
            axes[2].imshow(pc_img_rgb)
        else:
            axes[2].imshow(img_np)
        axes[2].set_title(f"Phase 2 PatchCore\n[{pc_res['status']}] Score: {pc_res['score']:.4f}")
        axes[2].axis("off")
        
        plt.tight_layout()
        save_file = out_dir / f"bottle_{defect_type}_{stem}_comparison.png"
        plt.savefig(save_file, dpi=150)
        plt.close(fig)
        
        # Copy individual Phase 1 and Phase 2 results into comparisons folder with clean names
        if ae_img is not None:
            cv2.imwrite(str(out_dir / f"bottle_{defect_type}_{stem}_phase1.png"), ae_img)
        if pc_img is not None:
            cv2.imwrite(str(out_dir / f"bottle_{defect_type}_{stem}_phase2.png"), pc_img)
            
        print(f"Saved visual comparison to: {save_file}")

if __name__ == "__main__":
    main()
