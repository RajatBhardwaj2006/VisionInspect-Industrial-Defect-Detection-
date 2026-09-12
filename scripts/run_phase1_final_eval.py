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

from src.models.autoencoder import Autoencoder
from src.data.dataset_loader import MVTecTestDataset
from src.detection.heatmap import compute_anomaly_map
from src.detection.localization import localize
from src.utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = Path("models/bottle/autoencoder.pth")
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        sys.exit(1)
        
    model = Autoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    dataset_root = "dataset/mvtec_anomaly_detection"
    category = "bottle"
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    image_scores = []
    image_labels = []
    
    all_data = []
    
    image_threshold = cfg["detection"]["image_threshold"]
    pixel_threshold = cfg["detection"]["pixel_threshold"]
    
    with torch.no_grad():
        for img, _, defect_type, is_defective, gt_mask in test_loader:
            img = img.to(device)
            recon = model(img)
            amap_tensor = compute_anomaly_map(img, recon).squeeze(0)
            amap = amap_tensor.cpu().numpy()
            
            img_score = float(amap.max())
            image_scores.append(img_score)
            is_def_val = 1 if is_defective.item() else 0
            image_labels.append(is_def_val)
            
            all_data.append({
                'defect_type': defect_type[0],
                'is_defective': is_def_val,
                'amap_tensor': amap_tensor.cpu(),
                'gt_mask': gt_mask.squeeze(0).squeeze(0).numpy() > 0,
                'img_score': img_score
            })
            
    # Compute Image AUROC
    image_auroc = roc_auc_score(image_labels, image_scores)
    
    # Threshold sweep
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    sweep_results = []
    
    best_f1 = -1
    best_thresh = pixel_threshold
    
    for thresh in thresholds:
        tp = fp = fn = 0
        for data in all_data:
            amap_t = data['amap_tensor']
            gt_binary = data['gt_mask']
            is_def_item = data['is_defective']
            img_score = data['img_score']
            
            regions = localize(amap_t, pixel_thresh=thresh)
            is_d = (img_score > image_threshold) and (len(regions) > 0)
            if is_d:
                top_reg = regions[0]
                if top_reg["total_mass"] < 25.0 and top_reg["score"] < 0.17:
                    is_d = False
                    
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
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        sweep_results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Best pixel threshold based on F1: {best_thresh} (F1: {best_f1:.4f})")
    
    # We will use the config pixel_threshold to generate per category metrics for consistency,
    # unless we want to update the config. Let's just use the best threshold and print it out.
    eval_thresh = cfg["detection"]["pixel_threshold"]
    
    # Per-category metrics
    cat_metrics = defaultdict(lambda: {'imgs': 0, 'detected': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    
    global_tp = global_fp = global_fn = 0
    
    for data in all_data:
        dtype = data['defect_type']
        amap_t = data['amap_tensor']
        gt_binary = data['gt_mask']
        is_def_item = data['is_defective']
        img_score = data['img_score']
        
        regions = localize(amap_t, pixel_thresh=eval_thresh)
        is_d = (img_score > image_threshold) and (len(regions) > 0)
        if is_d:
            top_reg = regions[0]
            if top_reg["total_mass"] < 25.0 and top_reg["score"] < 0.17:
                is_d = False
                
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
    
    print("\n--- Final Metrics ---")
    print(f"Image AUROC: {image_auroc:.4f}")
    print(f"Pixel Precision: {global_prec:.4f}")
    print(f"Pixel Recall: {global_rec:.4f}")
    print(f"Pixel F1: {global_f1:.4f}")
    print(f"Pixel IoU: {global_iou:.4f}")
    
    out_dir = Path("results/phase1_final")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. metrics.json
    metrics_dict = {
        "image_auroc": image_auroc,
        "pixel_precision": global_prec,
        "pixel_recall": global_rec,
        "pixel_f1": global_f1,
        "pixel_iou": global_iou,
        "best_pixel_threshold": best_thresh,
        "evaluated_threshold": eval_thresh
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
        writer.writerow(["threshold", "precision", "recall", "f1", "iou"])
        for s in sweep_results:
            writer.writerow([s["threshold"], s["precision"], s["recall"], s["f1"], s["iou"]])
            
    # 4. per_category.csv
    with open(out_dir / "per_category.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "images", "detected", "detection_rate", "precision", "recall", "f1", "iou"])
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

if __name__ == '__main__':
    main()
