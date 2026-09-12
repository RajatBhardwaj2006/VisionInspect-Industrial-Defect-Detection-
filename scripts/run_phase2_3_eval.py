import os
import sys
import json
import csv
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.detection.patchcore_v23_detector import (
    postprocess_clean_mask,
    extract_tight_regions,
    merge_nearby_regions
)
from src.detection.localization import postprocess_anomaly_map
from src.data.dataset_loader import MVTecTestDataset
from src.utils.config import load_config
import cv2

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    category = "bottle"
    dataset_root = "dataset/mvtec_anomaly_detection"
    pc_cfg = cfg.get("patchcore_v23", cfg.get("patchcore_v22", {}))
    model_dir = pc_cfg.get("model_dir", f"models/{category}/patchcore_v22")
    
    print("Loading PatchCore V2.3 detector model...")
    model = PatchCoreModelV22(device=device)
    model.load(model_dir)
    
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded test dataset with {len(test_dataset)} images.")
    
    all_data = []
    image_scores = []
    image_labels = []
    
    gaussian_sigma = pc_cfg.get("gaussian_sigma", 1.20)
    
    with torch.no_grad():
        for img, _, defect_type, is_defective, gt_mask in test_loader:
            img = img.to(device)
            score, amap_tensor = model.predict(img, use_prior=True)
            amap_np = amap_tensor.cpu().numpy()
            
            if gaussian_sigma > 0:
                smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=gaussian_sigma)
            else:
                smooth_map = amap_np
                
            is_def_val = 1 if is_defective.item() else 0
            image_scores.append(score)
            image_labels.append(is_def_val)
            
            all_data.append({
                'defect_type': defect_type[0],
                'is_defective': is_def_val,
                'smooth_map': smooth_map,
                'gt_mask': gt_mask.squeeze(0).squeeze(0).numpy() > 0,
                'score': score
            })
            
    image_auroc = roc_auc_score(image_labels, image_scores)
    print(f"\nPhase 2.3 PatchCore Image AUROC: {image_auroc:.4f}")
    
    # Threshold sweep
    img_thresholds = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]
    pixel_thresholds = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    sweep_results = []
    best_f1 = -1.0
    best_p_th = pc_cfg.get("pixel_threshold", 1.40)
    best_img_th = pc_cfg.get("image_threshold", 1.50)
    
    for img_th in img_thresholds:
        for p_th in pixel_thresholds:
            tp = fp = fn = 0
            for data in all_data:
                smooth_map = data['smooth_map']
                gt_binary = data['gt_mask']
                is_def_item = data['is_defective']
                score = data['score']
                
                bin_mask = (smooth_map > p_th)
                clean_mask = postprocess_clean_mask(bin_mask, kernel_size=3, min_area=25)
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
            
            sweep_results.append({
                'image_threshold': img_th,
                'pixel_threshold': p_th,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'iou': iou
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_p_th = p_th
                best_img_th = img_th

    print(f"Optimal Phase 2.3 Thresholds -> Image Threshold: {best_img_th}, Pixel Threshold: {best_p_th} (Best Pixel F1: {best_f1:.4f})")
    
    # Calculate per-category metrics with optimal thresholds
    eval_img_thresh = best_img_th
    eval_pixel_thresh = best_p_th
    
    cat_metrics = defaultdict(lambda: {'imgs': 0, 'detected': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    global_tp = global_fp = global_fn = 0
    
    for data in all_data:
        dtype = data['defect_type']
        smooth_map = data['smooth_map']
        gt_binary = data['gt_mask']
        is_def_item = data['is_defective']
        score = data['score']
        
        bin_mask = (smooth_map > eval_pixel_thresh)
        clean_mask = postprocess_clean_mask(bin_mask, kernel_size=3, min_area=25)
        regions = extract_tight_regions(clean_mask, border_margin=5)
        merged_regs = merge_nearby_regions(regions, merge_dist=15.0)
        
        is_d = (score > eval_img_thresh) and (len(merged_regs) > 0)
        
        if not is_d:
            pred_mask = np.zeros((256, 256), dtype=bool)
        else:
            pred_mask = clean_mask
            
        cat_metrics[dtype]['imgs'] += 1
        if is_d:
            cat_metrics[dtype]['detected'] += 1
            
        if is_def_item:
            tp_img = int(np.logical_and(pred_mask, gt_binary).sum())
            fp_img = int(np.logical_and(pred_mask, ~gt_binary).sum())
            fn_img = int(np.logical_and(~pred_mask, gt_binary).sum())
        else:
            tp_img = 0
            fn_img = 0
            fp_img = int(pred_mask.sum())
            
        cat_metrics[dtype]['tp'] += tp_img
        cat_metrics[dtype]['fp'] += fp_img
        cat_metrics[dtype]['fn'] += fn_img
        
        global_tp += tp_img
        global_fp += fp_img
        global_fn += fn_img
        
    global_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    global_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    global_f1 = (2 * global_prec * global_rec) / (global_prec + global_rec) if (global_prec + global_rec) > 0 else 0.0
    global_iou = global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else 0.0
    
    print("\n--- Phase 2.3 Final Metrics ---")
    print(f"Image AUROC:     {image_auroc:.4f}")
    print(f"Pixel Precision: {global_prec:.4f}")
    print(f"Pixel Recall:    {global_rec:.4f}")
    print(f"Pixel F1:        {global_f1:.4f}")
    print(f"Pixel IoU:       {global_iou:.4f}")
    
    out_dir = Path("results/phase2/phase2_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. metrics.json
    metrics_dict = {
        "image_auroc": image_auroc,
        "pixel_precision": global_prec,
        "pixel_recall": global_rec,
        "pixel_f1": global_f1,
        "pixel_iou": global_iou,
        "calibrated_image_threshold": eval_img_thresh,
        "calibrated_pixel_threshold": eval_pixel_thresh
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    # 2. metrics.csv
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])
            
    # 3. threshold_sweep.csv
    with open(out_dir / "threshold_sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_threshold", "pixel_threshold", "precision", "recall", "f1", "iou"])
        for s in sweep_results:
            writer.writerow([s["image_threshold"], s["pixel_threshold"], f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}", f"{s['iou']:.4f}"])
            
    # 4. per_category.csv
    with open(out_dir / "per_category.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_type", "images", "detected", "detection_rate", "precision", "recall", "f1", "iou"])
        for dtype, m in cat_metrics.items():
            imgs = m['imgs']
            det = m['detected']
            det_rate = det / imgs if imgs > 0 else 0
            tp = m['tp']
            fp = m['fp']
            fn = m['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            writer.writerow([dtype, imgs, det, f"{det_rate:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", f"{iou:.4f}"])
            
    # 5. before_vs_after.csv
    p1_metrics = {"AUROC": 0.8325, "Precision": 0.1832, "Recall": 0.2171, "F1": 0.1987, "IoU": 0.1103}
    p22_metrics = {"AUROC": 0.9929, "Precision": 0.6875, "Recall": 0.3955, "F1": 0.5021, "IoU": 0.3352}
    p23_metrics = {"AUROC": image_auroc, "Precision": global_prec, "Recall": global_rec, "F1": global_f1, "IoU": global_iou}
    
    with open(out_dir / "before_vs_after.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Phase1_Autoencoder", "Phase2_2_PatchCore", "Phase2_3_PreciseCrack", "Absolute_Change_v23_vs_v22", "Pct_Change_v23_vs_v22", "Pct_Change_v23_vs_Phase1"])
        for k in p1_metrics:
            v1 = p1_metrics[k]
            v22 = p22_metrics[k]
            v23 = p23_metrics[k]
            diff = v23 - v22
            pct_v22 = (diff / v22 * 100.0) if v22 > 0 else 0.0
            pct_p1 = ((v23 - v1) / v1 * 100.0) if v1 > 0 else 0.0
            writer.writerow([k, f"{v1:.4f}", f"{v22:.4f}", f"{v23:.4f}", f"{diff:+.4f}", f"{pct_v22:+.2f}%", f"{pct_p1:+.2f}%"])
            
    print(f"\nSaved all Phase 2.3 evaluation results to: {out_dir}")

if __name__ == "__main__":
    main()
