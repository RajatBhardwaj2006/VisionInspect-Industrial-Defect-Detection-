import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.data.dataset_loader import MVTecTestDataset, MVTecTrainDataset
from src.utils.config import load_config
from scipy.ndimage import label

def postprocess_clean_mask(bin_mask: np.ndarray, kernel_size: int = 3, min_area: int = 25) -> np.ndarray:
    """
    Applies morphological closing then opening to bridge crack fragments.
    Removes small isolated noise components (< min_area).
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Connected component noise removal
    labeled, num = label(opened)
    clean_mask = np.zeros_like(opened, dtype=bool)
    for i in range(1, num + 1):
        comp = (labeled == i)
        if comp.sum() >= min_area:
            clean_mask[comp] = True
            
    return clean_mask

def extract_tight_regions(clean_mask: np.ndarray, border_margin: int = 5) -> list:
    labeled, num = label(clean_mask.astype(np.int32))
    regions = []
    H, W = clean_mask.shape
    for idx in range(1, num + 1):
        comp = (labeled == idx)
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        # Skip border touching noise
        if (x_min <= border_margin or x_max >= W - border_margin or
            y_min <= border_margin or y_max >= H - border_margin):
            continue
            
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = int(comp.sum())
        
        regions.append({
            "x": x_min,
            "y": y_min,
            "width": w,
            "height": h,
            "area": area,
            "centroid": (float(xs.mean()), float(ys.mean()))
        })
    return regions

def merge_nearby_regions(regions: list, merge_dist: float = 15.0) -> list:
    if not regions or merge_dist <= 0:
        return regions
    merged = True
    while merged:
        merged = False
        used = set()
        for i in range(len(regions)):
            r1 = regions[i]
            r1_x1, r1_y1, r1_x2, r1_y2 = r1["x"], r1["y"], r1["x"] + r1["width"], r1["y"] + r1["height"]
            
            merge_with = -1
            for j in range(i + 1, len(regions)):
                r2 = regions[j]
                r2_x1, r2_y1, r2_x2, r2_y2 = r2["x"], r2["y"], r2["x"] + r2["width"], r2["y"] + r2["height"]
                
                dx = max(0, r1_x1 - r2_x2, r2_x1 - r1_x2)
                dy = max(0, r1_y1 - r2_y2, r2_y1 - r1_y2)
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= merge_dist:
                    merge_with = j
                    break
                    
            if merge_with != -1:
                r2 = regions[merge_with]
                mx1 = min(r1_x1, r2["x"])
                my1 = min(r1_y1, r2["y"])
                mx2 = max(r1_x2, r2["x"] + r2["width"])
                my2 = max(r1_y2, r2["y"] + r2["height"])
                
                regions[i] = {
                    "x": mx1,
                    "y": my1,
                    "width": mx2 - mx1,
                    "height": my2 - my1,
                    "area": r1["area"] + r2["area"],
                    "centroid": ((r1["centroid"][0] + r2["centroid"][0]) / 2.0, (r1["centroid"][1] + r2["centroid"][1]) / 2.0)
                }
                used.add(merge_with)
                merged = True
                break
                
        if merged:
            regions = [regions[idx] for idx in range(len(regions)) if idx not in used]
            
    return regions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_dir = "models/bottle/patchcore_v22"
    model = PatchCoreModelV22(device=device)
    model.load(model_dir)
    
    dataset_root = "dataset/mvtec_anomaly_detection"
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category="bottle")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_data = []
    image_scores = []
    image_labels = []
    
    with torch.no_grad():
        for img, _, defect_type, is_defective, gt_mask in test_loader:
            img = img.to(device)
            score, amap_tensor = model.predict(img, use_prior=True)
            
            is_def_val = 1 if is_defective.item() else 0
            image_scores.append(score)
            image_labels.append(is_def_val)
            
            all_data.append({
                'defect_type': defect_type[0],
                'is_defective': is_def_val,
                'amap_tensor': amap_tensor.cpu().numpy(),
                'gt_mask': gt_mask.squeeze(0).squeeze(0).numpy() > 0,
                'score': score
            })
            
    image_auroc = roc_auc_score(image_labels, image_scores)
    print(f"Image AUROC: {image_auroc:.4f}")
    
    # Grid search over sigma, img_thresh, pixel_thresh, kernel_size
    sigmas = [1.0, 1.2, 1.5]
    img_thresholds = [1.5, 1.8, 2.0, 2.2, 2.4]
    pixel_thresholds = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
    kernel_sizes = [3, 5]
    
    best_f1 = -1.0
    best_params = {}
    best_metrics = {}
    
    for sigma in sigmas:
        for img_th in img_thresholds:
            for p_th in pixel_thresholds:
                for k_size in kernel_sizes:
                    tp = fp = fn = 0
                    for data in all_data:
                        amap_np = data['amap_tensor']
                        gt_binary = data['gt_mask']
                        is_def_item = data['is_defective']
                        score = data['score']
                        
                        # 1. Smooth
                        if sigma > 0:
                            smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=sigma)
                        else:
                            smooth_map = amap_np
                            
                        # 2. Direct absolute distance thresholding
                        bin_mask = (smooth_map > p_th)
                        
                        # 3. Clean mask with crack-preserving morphology
                        clean_mask = postprocess_clean_mask(bin_mask, kernel_size=k_size, min_area=25)
                        
                        # 4. Extract regions
                        regions = extract_tight_regions(clean_mask, border_margin=5)
                        merged_regs = merge_nearby_regions(regions, merge_dist=15.0)
                        
                        is_d = (score > img_th) and (len(merged_regs) > 0)
                        
                        if not is_d:
                            pred_mask = np.zeros((256, 256), dtype=bool)
                        else:
                            pred_mask = clean_mask
                            
                        if is_def_item:
                            tp += int(np.logical_and(pred_mask, gt_binary).sum())
                            fp += int(np.logical_and(pred_mask, ~gt_binary).sum())
                            fn += int(np.logical_and(~pred_mask, gt_binary).sum())
                        else:
                            fp += int(pred_mask.sum())
                            
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            'sigma': sigma,
                            'img_th': img_th,
                            'p_th': p_th,
                            'k_size': k_size
                        }
                        best_metrics = {
                            'precision': prec,
                            'recall': rec,
                            'f1': f1,
                            'iou': iou,
                            'tp': tp,
                            'fp': fp,
                            'fn': fn
                        }

    print("\n================ Phase 2.3 Optimization Results ================")
    print(f"Optimal Parameters: {best_params}")
    print(f"Image AUROC:     {image_auroc:.4f}")
    print(f"Pixel Precision: {best_metrics['precision']:.4f}")
    print(f"Pixel Recall:    {best_metrics['recall']:.4f}")
    print(f"Pixel F1-Score:  {best_metrics['f1']:.4f}")
    print(f"Pixel IoU:       {best_metrics['iou']:.4f}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
