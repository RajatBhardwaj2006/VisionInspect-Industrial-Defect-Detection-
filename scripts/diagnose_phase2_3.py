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

from src.models.patchcore_v22 import PatchCoreModelV22
from src.detection.localization import localize, postprocess_anomaly_map
from src.utils.config import load_config
from torch.utils.data import DataLoader
from src.data.dataset_loader import MVTecTestDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cfg = load_config()
    pc_cfg = cfg.get("patchcore_v22", {})
    model_dir = pc_cfg.get("model_dir", "models/bottle/patchcore_v22")
    
    model = PatchCoreModelV22(device=device)
    model.load(model_dir)
    
    dataset_root = "dataset/mvtec_anomaly_detection"
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category="bottle")
    
    out_dir = Path("results/phase2_3_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        ("good", "001.png"),
        ("broken_large", "000.png"),
        ("broken_small", "000.png"),
        ("contamination", "000.png"),
    ]
    
    gaussian_sigma = pc_cfg.get("gaussian_sigma", 1.50)
    pixel_thresh = pc_cfg.get("pixel_threshold", 0.60)
    image_thresh = pc_cfg.get("image_threshold", 2.20)
    
    for defect_type, img_name in targets:
        img_path = Path(dataset_root) / "bottle" / "test" / defect_type / img_name
        gt_path = Path(dataset_root) / "bottle" / "ground_truth" / defect_type / img_name.replace(".png", "_mask.png")
        has_gt = gt_path.exists()
        
        img_pil = Image.open(img_path).convert("RGB")
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Raw distance map
            raw_score, raw_amap = model._predict_raw(img_tensor)
            # Prior subtracted map
            sub_score, sub_amap = model.predict(img_tensor, use_prior=True)
            
        raw_amap_np = raw_amap.cpu().numpy()
        sub_amap_np = sub_amap.cpu().numpy()
        prior_np = model.normal_spatial_prior.cpu().numpy() if model.normal_spatial_prior is not None else np.zeros((256, 256))
        
        # Smoothed sub map
        smooth_sub = cv2.GaussianBlur(sub_amap_np, (0, 0), sigmaX=gaussian_sigma)
        
        # Current localization logic:
        # Min-Max normalized smooth map
        norm_map = (smooth_sub - smooth_sub.min()) / (smooth_sub.max() - smooth_sub.min() + 1e-8)
        bin_mask_norm = (norm_map > pixel_thresh).astype(np.uint8)
        
        # Direct distance thresholding (without min-max rescale)
        bin_mask_direct = (smooth_sub > 1.2).astype(np.uint8)
        
        # Regions from localization
        regions = localize(torch.from_numpy(smooth_sub).to(device), pixel_thresh=pixel_thresh)
        
        # Bounding box filled mask vs pixel mask
        bbox_mask = np.zeros((256, 256), dtype=np.uint8)
        pixel_region_mask = np.zeros((256, 256), dtype=np.uint8)
        
        for r in regions:
            x, y, w, h = r["x"], r["y"], r["width"], r["height"]
            bbox_mask[y:y+h, x:x+w] = 1
            # Pixel region mask (intersection of bin_mask_norm and bbox)
            pixel_region_mask[y:y+h, x:x+w] = bin_mask_norm[y:y+h, x:x+w]
            
        # GT mask
        if has_gt:
            gt_pil = Image.open(gt_path).convert("L")
            gt_np = (cv2.resize(np.array(gt_pil), (256, 256)) > 0).astype(np.uint8)
        else:
            gt_np = np.zeros((256, 256), dtype=np.uint8)
            
        # Compute GT overlap metrics for Bbox mask vs Pixel region mask vs Direct threshold mask
        def calc_m(pred, gt):
            tp = np.logical_and(pred, gt).sum()
            fp = np.logical_and(pred, ~gt).sum()
            fn = np.logical_and(~pred, gt).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = (2*p*r)/(p+r) if (p+r) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            return p, r, f, iou, int(tp), int(fp), int(fn)

        p_bbox, r_bbox, f_bbox, iou_bbox, tp_b, fp_b, fn_b = calc_m(bbox_mask, gt_np)
        p_pix, r_pix, f_pix, iou_pix, tp_p, fp_p, fn_p = calc_m(pixel_region_mask, gt_np)
        p_dir, r_dir, f_dir, iou_dir, tp_d, fp_d, fn_d = calc_m(bin_mask_direct, gt_np)
        
        print(f"\n--- Diagnostic: {defect_type}/{img_name} ---")
        print(f"Raw Max: {raw_amap_np.max():.4f}, Subtracted Max: {sub_amap_np.max():.4f}, GT Area: {gt_np.sum()}")
        print(f"  [Bbox Fill]        P: {p_bbox:.4f}, R: {r_bbox:.4f}, F1: {f_bbox:.4f}, IoU: {iou_bbox:.4f} (TP: {tp_b}, FP: {fp_b}, FN: {fn_b})")
        print(f"  [Pixel Region]     P: {p_pix:.4f}, R: {r_pix:.4f}, F1: {f_pix:.4f}, IoU: {iou_pix:.4f} (TP: {tp_p}, FP: {fp_p}, FN: {fn_p})")
        print(f"  [Direct Threshold] P: {p_dir:.4f}, R: {r_dir:.4f}, F1: {f_dir:.4f}, IoU: {iou_dir:.4f} (TP: {tp_d}, FP: {fp_d}, FN: {fn_d})")
        
        # Plot 6-panel diagnostic
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        axes[0, 0].imshow(img_pil)
        axes[0, 0].set_title(f"Original ({defect_type}/{img_name})")
        axes[0, 0].axis("off")
        
        im1 = axes[0, 1].imshow(raw_amap_np, cmap="jet")
        axes[0, 1].set_title(f"Raw Anomaly Map (Max: {raw_amap_np.max():.2f})")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[0, 2].imshow(sub_amap_np, cmap="jet")
        axes[0, 2].set_title(f"Cleaned Map (Prior Subtracted)")
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2])
        
        axes[1, 0].imshow(bin_mask_norm, cmap="gray")
        axes[1, 0].set_title(f"Threshold Mask (Rescaled > {pixel_thresh})")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(pixel_region_mask, cmap="gray")
        axes[1, 1].set_title(f"Region Mask (Pixel F1: {f_pix:.2f})")
        axes[1, 1].axis("off")
        
        # Overlay on original image
        overlay = np.array(img_pil).copy()
        if regions:
            for r in regions:
                bx, by, bw, bh = r["x"], r["y"], r["width"], r["height"]
                cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title(f"Overlay ({len(regions)} Regions)")
        axes[1, 2].axis("off")
        
        plt.tight_layout()
        save_path = out_dir / f"diag_phase2_3_{defect_type}_{img_name.split('.')[0]}.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved diagnostic figure to: {save_path}")

if __name__ == "__main__":
    main()
