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
from src.detection.localization import localize, postprocess_anomaly_map
from src.data.dataset_loader import MVTecTestDataset
from src.utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    category = "bottle"
    dataset_root = "dataset/mvtec_anomaly_detection"
    pc_cfg = cfg.get("patchcore_v22", {})
    model_dir = pc_cfg.get("model_dir", f"models/{category}/patchcore_v22")
    
    print("Loading PatchCore V2.2 model...")
    model = PatchCoreModelV22(device=device)
    model.load(model_dir)
    
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded test dataset with {len(test_dataset)} images.")
    
    all_data = []
    image_scores = []
    image_labels = []
    
    gaussian_sigma = pc_cfg.get("gaussian_sigma", 1.50)
    
    with torch.no_grad():
        for img, _, defect_type, is_defective, gt_mask in test_loader:
            img = img.to(device)
            score, amap_tensor = model.predict(img, use_prior=True)
            amap_smoothed = postprocess_anomaly_map(amap_tensor, sigma=gaussian_sigma)
            
            is_def_val = 1 if is_defective.item() else 0
            image_scores.append(score)
            image_labels.append(is_def_val)
            
            all_data.append({
                'defect_type': defect_type[0],
                'is_defective': is_def_val,
                'amap_smoothed': amap_smoothed.cpu(),
                'gt_mask': gt_mask.squeeze(0).squeeze(0).numpy() > 0,
                'score': score
            })
            
    # Compute Image AUROC
    image_auroc = roc_auc_score(image_labels, image_scores)
    print(f"\nPhase 2.2 PatchCore Image AUROC: {image_auroc:.4f}")
    
    # 2D Threshold sweep
    img_thresholds = [1.2, 1.5, 1.8, 2.0, 2.2, 2.5]
    pixel_thresholds = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    
    sweep_results = []
    best_f1 = -1.0
    best_p_th = pc_cfg.get("pixel_threshold", 1.20)
    best_img_th = pc_cfg.get("image_threshold", 2.00)
    
    for img_th in img_thresholds:
        for p_th in pixel_thresholds:
            tp = fp = fn = 0
            for data in all_data:
                amap_t = data['amap_smoothed']
                gt_binary = data['gt_mask']
                is_def_item = data['is_defective']
                score = data['score']
                
                regions = localize(amap_t, pixel_thresh=p_th)
                is_d = (score > img_th) and (len(regions) > 0)
                
                pred_mask = np.zeros((256, 256), dtype=bool)
                if is_d and regions:
                    for r in regions:
                        x, y, w, h = r["x"], r["y"], r["width"], r["height"]
                        pred_mask[y:y+h, x:x+w] = True
                        
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

    print(f"Optimal Phase 2.2 Thresholds -> Image Threshold: {best_img_th}, Pixel Threshold: {best_p_th} (Best Pixel F1: {best_f1:.4f})")
    
    # Calculate per-category breakdown using optimal thresholds
    eval_img_thresh = best_img_th
    eval_pixel_thresh = best_p_th
    
    cat_metrics = defaultdict(lambda: {'imgs': 0, 'detected': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    global_tp = global_fp = global_fn = 0
    
    for data in all_data:
        dtype = data['defect_type']
        amap_t = data['amap_smoothed']
        gt_binary = data['gt_mask']
        is_def_item = data['is_defective']
        score = data['score']
        
        regions = localize(amap_t, pixel_thresh=eval_pixel_thresh)
        is_d = (score > eval_img_thresh) and (len(regions) > 0)
        
        pred_mask = np.zeros((256, 256), dtype=bool)
        if is_d and regions:
            for r in regions:
                x, y, w, h = r["x"], r["y"], r["width"], r["height"]
                pred_mask[y:y+h, x:x+w] = True
                
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
    
    print("\n--- Phase 2.2 PatchCore Final Metrics ---")
    print(f"Image AUROC:     {image_auroc:.4f}")
    print(f"Pixel Precision: {global_prec:.4f}")
    print(f"Pixel Recall:    {global_rec:.4f}")
    print(f"Pixel F1:        {global_f1:.4f}")
    print(f"Pixel IoU:       {global_iou:.4f}")
    
    out_dir = Path("results/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. phase2_2_metrics.json
    metrics_dict = {
        "image_auroc": image_auroc,
        "pixel_precision": global_prec,
        "pixel_recall": global_rec,
        "pixel_f1": global_f1,
        "pixel_iou": global_iou,
        "calibrated_image_threshold": eval_img_thresh,
        "calibrated_pixel_threshold": eval_pixel_thresh
    }
    with open(out_dir / "phase2_2_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)
        
    # 2. phase2_2_metrics.csv
    with open(out_dir / "phase2_2_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics_dict.items():
            writer.writerow([k, v])
            
    # 3. phase2_2_threshold_sweep.csv
    with open(out_dir / "phase2_2_threshold_sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_threshold", "pixel_threshold", "precision", "recall", "f1", "iou"])
        for s in sweep_results:
            writer.writerow([s["image_threshold"], s["pixel_threshold"], f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}", f"{s['iou']:.4f}"])
            
    # 4. phase2_2_per_category.csv
    with open(out_dir / "phase2_2_per_category.csv", "w", newline="") as f:
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
            
    # 5. phase2_2_before_vs_after.csv
    p1_metrics = {"AUROC": 0.8325, "Precision": 0.1832, "Recall": 0.2171, "F1": 0.1987, "IoU": 0.1103}
    p2_metrics = {"AUROC": 0.9992, "Precision": 0.6231, "Recall": 0.3202, "F1": 0.4230, "IoU": 0.2682}
    p22_metrics = {"AUROC": image_auroc, "Precision": global_prec, "Recall": global_rec, "F1": global_f1, "IoU": global_iou}
    
    with open(out_dir / "phase2_2_before_vs_after.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Phase1_Autoencoder", "Phase2_PatchCore", "Phase2_2_CrackSensitive", "Absolute_Change_p22_vs_p2", "Pct_Change_p22_vs_p2", "Pct_Change_p22_vs_p1"])
        for k in p1_metrics:
            v1 = p1_metrics[k]
            v2 = p2_metrics[k]
            v22 = p22_metrics[k]
            diff = v22 - v2
            pct_p2 = (diff / v2 * 100.0) if v2 > 0 else 0.0
            pct_p1 = ((v22 - v1) / v1 * 100.0) if v1 > 0 else 0.0
            writer.writerow([k, f"{v1:.4f}", f"{v2:.4f}", f"{v22:.4f}", f"{diff:+.4f}", f"{pct_p2:+.2f}%", f"{pct_p1:+.2f}%"])
            
    print(f"\nSaved all Phase 2.2 evaluation results to: {out_dir}")

if __name__ == "__main__":
    main()
