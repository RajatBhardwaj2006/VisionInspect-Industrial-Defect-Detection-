import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.detection.patchcore_v23_detector import (
    postprocess_clean_mask,
    extract_tight_regions,
    merge_nearby_regions
)
from src.data.dataset_loader import MVTecTestDataset
from src.utils.config import load_config, get_category_config

def evaluate_single_category(
    category: str,
    dataset_root: str = "dataset/mvtec_anomaly_detection",
    device: torch.device = None,
    calibrate_thresholds: bool = True
) -> Dict[str, Any]:
    category = category.strip().lower()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory for category '{category}' not found at {model_dir}. "
            f"Train it first via: python scripts/train_patchcore.py --category {category}"
        )
        
    print(f"\n========================================================")
    print(f"   EVALUATING CATEGORY: {category.upper()}")
    print(f"========================================================")
    print(f"Device:              {device}")
    print(f"Model Directory:     {model_dir}")
    print(f"Use Spatial Prior:   {cat_cfg['use_spatial_prior']}")
    
    model = PatchCoreModelV22(device=device)
    model.load(str(model_dir))
    
    # Metadata inspection
    use_prior = cat_cfg["use_spatial_prior"]
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                if "use_spatial_prior" in meta:
                    use_prior = bool(meta["use_spatial_prior"])
        except Exception:
            pass
            
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples for '{category}'.")
    
    gaussian_sigma = cat_cfg.get("gaussian_sigma", 1.20)
    morph_kernel = cat_cfg.get("morphology_kernel", 3)
    min_area = cat_cfg.get("min_region_area", 25)
    border_margin = cat_cfg.get("border_margin", 5)
    merge_dist = cat_cfg.get("merge_distance", 15.0)
    
    all_data = []
    image_scores = []
    image_labels = []
    normal_scores = []
    defective_scores = []
    
    print("Inference on test images...")
    with torch.no_grad():
        for idx, (img, _, defect_type, is_defective, gt_mask) in enumerate(test_loader):
            img = img.to(device)
            score, amap_tensor = model.predict(img, use_prior=use_prior)
            amap_np = amap_tensor.cpu().numpy()
            
            if gaussian_sigma > 0:
                smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=gaussian_sigma)
            else:
                smooth_map = amap_np
                
            is_def_val = 1 if is_defective.item() else 0
            image_scores.append(score)
            image_labels.append(is_def_val)
            
            if is_def_val:
                defective_scores.append(score)
            else:
                normal_scores.append(score)
                
            all_data.append({
                'defect_type': defect_type[0],
                'is_defective': is_def_val,
                'smooth_map': smooth_map,
                'gt_mask': gt_mask.squeeze(0).squeeze(0).numpy() > 0,
                'score': score
            })
            
    # Calculate Image AUROC
    try:
        image_auroc = float(roc_auc_score(image_labels, image_scores))
    except Exception as e:
        print(f"Warning: Could not compute AUROC: {e}")
        image_auroc = 0.50
        
    print(f"\n[Raw Distribution] Normal Scores: min={min(normal_scores):.4f}, max={max(normal_scores):.4f}, mean={np.mean(normal_scores):.4f}")
    print(f"[Raw Distribution] Defective Scores: min={min(defective_scores):.4f}, max={max(defective_scores):.4f}, mean={np.mean(defective_scores):.4f}")
    print(f"Image AUROC: {image_auroc:.4f}")
    
    # 2D Threshold Sweep
    sweep_results = []
    
    if calibrate_thresholds:
        print("\nPerforming 2D threshold sweep for optimal Image & Pixel thresholds...")
        # Determine candidate grid based on empirical distributions
        all_smooth_maxes = [d['smooth_map'].max() for d in all_data]
        all_smooth_means = [d['smooth_map'].mean() for d in all_data]
        
        min_img_s = min(normal_scores)
        max_img_s = max(image_scores)
        
        # Grid for image threshold
        img_thresholds = np.linspace(
            np.percentile(normal_scores, 60),
            np.percentile(defective_scores, 80) if len(defective_scores) > 0 else max_img_s,
            8
        ).tolist()
        # Add configured threshold as well
        img_thresholds.append(cat_cfg.get("image_threshold", 1.50))
        img_thresholds = sorted(list(set([round(t, 2) for t in img_thresholds if t > 0])))
        
        # Grid for pixel threshold
        p_min = max(0.5, float(np.percentile(all_smooth_means, 50)))
        p_max = float(np.percentile(all_smooth_maxes, 85))
        pixel_thresholds = np.linspace(p_min, p_max, 8).tolist()
        pixel_thresholds.append(cat_cfg.get("pixel_threshold", 1.40))
        pixel_thresholds = sorted(list(set([round(p, 2) for p in pixel_thresholds if p > 0])))
        
        best_f1 = -1.0
        best_img_th = cat_cfg.get("image_threshold", 1.50)
        best_pixel_th = cat_cfg.get("pixel_threshold", 1.40)
        
        for img_th in img_thresholds:
            for p_th in pixel_thresholds:
                tp = fp = fn = 0
                for data in all_data:
                    smooth_map = data['smooth_map']
                    gt_binary = data['gt_mask']
                    is_def_item = data['is_defective']
                    score = data['score']
                    
                    bin_mask = (smooth_map > p_th)
                    clean_mask = postprocess_clean_mask(bin_mask, kernel_size=morph_kernel, min_area=min_area)
                    regions = extract_tight_regions(clean_mask, border_margin=border_margin)
                    merged_regs = merge_nearby_regions(regions, merge_dist=merge_dist)
                    
                    is_d = (score > img_th) and (len(merged_regs) > 0)
                    
                    pred_mask = clean_mask if is_d else np.zeros_like(clean_mask, dtype=bool)
                    
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
                    best_img_th = img_th
                    best_pixel_th = p_th
                    
        calibrated_img_th = best_img_th
        calibrated_pixel_th = best_pixel_th
        print(f"Optimal Calibrated Thresholds: Image Threshold = {calibrated_img_th}, Pixel Threshold = {calibrated_pixel_th} (Peak F1: {best_f1:.4f})")
    else:
        calibrated_img_th = cat_cfg.get("image_threshold", 1.50)
        calibrated_pixel_th = cat_cfg.get("pixel_threshold", 1.40)
        
    # Evaluate final metrics at chosen thresholds
    cat_metrics = defaultdict(lambda: {'imgs': 0, 'detected': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    global_tp = global_fp = global_fn = 0
    normal_fp_count = 0
    
    for data in all_data:
        dtype = data['defect_type']
        smooth_map = data['smooth_map']
        gt_binary = data['gt_mask']
        is_def_item = data['is_defective']
        score = data['score']
        
        bin_mask = (smooth_map > calibrated_pixel_th)
        clean_mask = postprocess_clean_mask(bin_mask, kernel_size=morph_kernel, min_area=min_area)
        regions = extract_tight_regions(clean_mask, border_margin=border_margin)
        merged_regs = merge_nearby_regions(regions, merge_dist=merge_dist)
        
        is_d = (score > calibrated_img_th) and (len(merged_regs) > 0)
        pred_mask = clean_mask if is_d else np.zeros_like(clean_mask, dtype=bool)
        
        cat_metrics[dtype]['imgs'] += 1
        if is_d:
            cat_metrics[dtype]['detected'] += 1
            if not is_def_item:
                normal_fp_count += 1
                
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
        
    final_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    final_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    final_f1 = (2 * final_prec * final_rec) / (final_prec + final_rec) if (final_prec + final_rec) > 0 else 0.0
    final_iou = global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else 0.0
    
    total_samples = len(all_data)
    total_defective = len(defective_scores)
    total_normal = len(normal_scores)
    detected_defective = sum(cat_metrics[d]['detected'] for d in cat_metrics if d != 'good')
    detection_rate = detected_defective / total_defective if total_defective > 0 else 0.0
    normal_accuracy = (total_normal - normal_fp_count) / total_normal if total_normal > 0 else 1.0
    
    print(f"\n---------------- FINAL RESULTS FOR '{category.upper()}' ----------------")
    print(f"Total Test Images:     {total_samples} (Normal: {total_normal}, Defective: {total_defective})")
    print(f"Defect Detection Rate: {detected_defective}/{total_defective} ({detection_rate*100:.1f}%)")
    print(f"Normal Image Accuracy: {total_normal - normal_fp_count}/{total_normal} ({normal_accuracy*100:.1f}%)")
    print(f"Image AUROC:           {image_auroc:.4f}")
    print(f"Pixel Precision:       {final_prec:.4f}")
    print(f"Pixel Recall:          {final_rec:.4f}")
    print(f"Pixel F1-Score:        {final_f1:.4f}")
    print(f"Pixel IoU:             {final_iou:.4f}")
    print(f"-------------------------------------------------------------")
    
    # Save results under results/phase3/<category>/
    out_dir = Path("results/phase3") / category
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        "category": category,
        "image_auroc": image_auroc,
        "pixel_precision": final_prec,
        "pixel_recall": final_rec,
        "pixel_f1": final_f1,
        "pixel_iou": final_iou,
        "calibrated_image_threshold": calibrated_img_th,
        "calibrated_pixel_threshold": calibrated_pixel_th,
        "total_test_images": total_samples,
        "normal_images": total_normal,
        "defective_images": total_defective,
        "detected_defective": detected_defective,
        "detection_rate": detection_rate,
        "normal_false_positives": normal_fp_count,
        "normal_accuracy": normal_accuracy,
        "tp_pixels": global_tp,
        "fp_pixels": global_fp,
        "fn_pixels": global_fn,
        "use_spatial_prior": use_prior
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results_summary, f, indent=4)
        
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in results_summary.items():
            writer.writerow([k, v])
            
    with open(out_dir / "per_category.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_type", "images", "detected", "detection_rate", "precision", "recall", "f1", "iou"])
        for dtype, m in sorted(cat_metrics.items()):
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
            
    if sweep_results:
        with open(out_dir / "threshold_sweep.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_threshold", "pixel_threshold", "precision", "recall", "f1", "iou"])
            for s in sweep_results:
                writer.writerow([s["image_threshold"], s["pixel_threshold"], f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}", f"{s['iou']:.4f}"])
                
    print(f"Saved evaluation artifacts to: {out_dir}")
    return results_summary

def update_multi_category_summary(summaries: List[Dict[str, Any]]):
    out_file = Path("results/phase3/multi_category_summary.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing summaries if present so we can update/merge
    existing = {}
    if out_file.exists():
        try:
            with open(out_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing[row["category"]] = row
        except Exception:
            pass
            
    for s in summaries:
        cat = s["category"]
        existing[cat] = {
            "category": cat,
            "image_auroc": f"{s['image_auroc']:.4f}",
            "pixel_precision": f"{s['pixel_precision']:.4f}",
            "pixel_recall": f"{s['pixel_recall']:.4f}",
            "pixel_f1": f"{s['pixel_f1']:.4f}",
            "pixel_iou": f"{s['pixel_iou']:.4f}",
            "detection_rate": f"{s['detection_rate']*100:.1f}%",
            "normal_accuracy": f"{s['normal_accuracy']*100:.1f}%",
            "image_threshold": f"{s['calibrated_image_threshold']:.2f}",
            "pixel_threshold": f"{s['calibrated_pixel_threshold']:.2f}",
            "use_spatial_prior": s["use_spatial_prior"],
            "total_test_images": s["total_test_images"]
        }
        
    headers = [
        "category", "image_auroc", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou",
        "detection_rate", "normal_accuracy", "image_threshold", "pixel_threshold",
        "use_spatial_prior", "total_test_images"
    ]
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for cat in sorted(existing.keys()):
            writer.writerow(existing[cat])
            
    print(f"\nMulti-Category Summary updated: {out_file.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="VisionInspect - Multi-Category Evaluation Suite")
    parser.add_argument("--category", type=str, required=True, help="Category name (bottle, leather, transistor, zipper, screw) or 'all'")
    parser.add_argument("--no-calibrate", action="store_true", help="Use current config thresholds without re-calibrating sweep")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_categories = ['bottle', 'leather', 'transistor', 'zipper', 'screw']
    
    if args.category.lower() == "all":
        categories_to_run = target_categories
    else:
        categories_to_run = [args.category.lower()]
        
    all_summaries = []
    for cat in categories_to_run:
        try:
            summary = evaluate_single_category(
                category=cat,
                device=device,
                calibrate_thresholds=(not args.no_calibrate)
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"Error evaluating category '{cat}': {e}", file=sys.stderr)
            
    if all_summaries:
        update_multi_category_summary(all_summaries)

if __name__ == "__main__":
    main()
