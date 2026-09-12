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

def evaluate_unbiased_category(
    category: str,
    locked_params: Dict[str, Any],
    dataset_root: str = "dataset/mvtec_anomaly_detection",
    device: torch.device = None,
    out_dir: Path = None
) -> Dict[str, Any]:
    """
    Evaluates final MVTec test set using strictly LOCKED parameters from validation calibration.
    Zero tuning or threshold sweeps are performed on the test set.
    """
    category = category.strip().lower()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory for '{category}' not found at {model_dir}")
        
    img_th = locked_params.get("locked_image_threshold", cat_cfg.get("image_threshold", 1.50))
    pixel_th = locked_params.get("locked_pixel_threshold", cat_cfg.get("pixel_threshold", 1.40))
    use_prior = locked_params.get("use_spatial_prior", cat_cfg.get("use_spatial_prior", True))
    gaussian_sigma = locked_params.get("locked_gaussian_sigma", cat_cfg.get("gaussian_sigma", 1.20))
    morph_kernel = locked_params.get("locked_morphology_kernel", cat_cfg.get("morphology_kernel", 3))
    min_area = locked_params.get("locked_min_region_area", cat_cfg.get("min_region_area", 25))
    border_margin = locked_params.get("locked_border_margin", cat_cfg.get("border_margin", 5))
    merge_dist = locked_params.get("locked_merge_distance", cat_cfg.get("merge_distance", 15.0))
    
    print(f"\n========================================================")
    print(f"   FINAL UNBIASED TEST EVALUATION: {category.upper()}")
    print(f"========================================================")
    print(f"Device:                 {device}")
    print(f"Model Directory:        {model_dir}")
    print(f"Spatial Prior:          {use_prior}")
    print(f"LOCKED Image Threshold: {img_th}")
    print(f"LOCKED Pixel Threshold: {pixel_th}")
    print(f"LOCKED Gaussian Sigma:  {gaussian_sigma}")
    
    model = PatchCoreModelV22(device=device)
    model.load(str(model_dir))
    
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(test_dataset)} test samples for '{category}'.")
    
    cat_metrics = defaultdict(lambda: {'imgs': 0, 'detected': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    image_scores = []
    image_labels = []
    normal_scores = []
    defective_scores = []
    normal_fp_count = 0
    global_tp = global_fp = global_fn = 0
    
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
            dtype = defect_type[0]
            gt_binary = gt_mask.squeeze(0).squeeze(0).numpy() > 0
            
            image_scores.append(score)
            image_labels.append(is_def_val)
            if is_def_val:
                defective_scores.append(score)
            else:
                normal_scores.append(score)
                
            # Apply locked thresholds
            bin_mask = (smooth_map > pixel_th)
            clean_mask = postprocess_clean_mask(bin_mask, kernel_size=morph_kernel, min_area=min_area)
            regions = extract_tight_regions(clean_mask, border_margin=border_margin)
            merged_regs = merge_nearby_regions(regions, merge_dist=merge_dist)
            
            is_pred_defective = (score > img_th) and (len(merged_regs) > 0)
            pred_mask = clean_mask if is_pred_defective else np.zeros_like(clean_mask, dtype=bool)
            
            cat_metrics[dtype]['imgs'] += 1
            if is_pred_defective:
                cat_metrics[dtype]['detected'] += 1
                if not is_def_val:
                    normal_fp_count += 1
                    
            if is_def_val:
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
            
    # Calculate metrics
    try:
        image_auroc = float(roc_auc_score(image_labels, image_scores))
    except Exception:
        image_auroc = 0.50
        
    prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else 0.0
    
    total_samples = len(image_scores)
    total_normal = len(normal_scores)
    total_defective = len(defective_scores)
    detected_defective = sum(cat_metrics[d]['detected'] for d in cat_metrics if d != 'good')
    detection_rate = detected_defective / total_defective if total_defective > 0 else 0.0
    normal_accuracy = (total_normal - normal_fp_count) / total_normal if total_normal > 0 else 1.0
    
    print(f"\n---------------- UNBIASED BENCHMARK FOR '{category.upper()}' ----------------")
    print(f"Total Test Images:     {total_samples} (Normal: {total_normal}, Defective: {total_defective})")
    print(f"Defect Detection Rate: {detected_defective}/{total_defective} ({detection_rate*100:.1f}%)")
    print(f"Normal Image Accuracy: {total_normal - normal_fp_count}/{total_normal} ({normal_accuracy*100:.1f}%)")
    print(f"Image AUROC:           {image_auroc:.4f}")
    print(f"Pixel Precision:       {prec:.4f}")
    print(f"Pixel Recall:          {rec:.4f}")
    print(f"Pixel F1-Score:        {f1:.4f}")
    print(f"Pixel IoU:             {iou:.4f}")
    print(f"-------------------------------------------------------------")
    
    cat_out = out_dir / category
    cat_out.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        "category": category,
        "evaluation_type": "unbiased_locked_test",
        "image_auroc": round(image_auroc, 4),
        "pixel_precision": round(prec, 4),
        "pixel_recall": round(rec, 4),
        "pixel_f1": round(f1, 4),
        "pixel_iou": round(iou, 4),
        "detection_rate": round(detection_rate, 4),
        "normal_accuracy": round(normal_accuracy, 4),
        "locked_image_threshold": img_th,
        "locked_pixel_threshold": pixel_th,
        "total_test_images": total_samples,
        "normal_images": total_normal,
        "defective_images": total_defective,
        "detected_defective": detected_defective,
        "normal_false_positives": normal_fp_count,
        "use_spatial_prior": use_prior
    }
    
    with open(cat_out / "metrics.json", "w") as f:
        json.dump(results_summary, f, indent=4)
        
    with open(cat_out / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in results_summary.items():
            writer.writerow([k, v])
            
    with open(cat_out / "per_category.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_type", "images", "detected", "detection_rate", "precision", "recall", "f1", "iou"])
        for dtype, m in sorted(cat_metrics.items()):
            imgs = m['imgs']
            det = m['detected']
            det_r = det / imgs if imgs > 0 else 0
            tp = m['tp']
            fp = m['fp']
            fn = m['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            iu = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            writer.writerow([dtype, imgs, det, f"{det_r:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", f"{iu:.4f}"])
            
    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Unbiased Evaluation with Locked Calibration Thresholds")
    parser.add_argument("--categories", nargs="+", default=['bottle', 'leather', 'transistor', 'zipper', 'screw'])
    parser.add_argument("--dataset-root", type=str, default="dataset/mvtec_anomaly_detection")
    parser.add_argument("--locked-file", type=str, default="results/phase3/final_validation_locked/locked_thresholds.json")
    parser.add_argument("--output-dir", type=str, default="results/phase3/final_test")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    locked_data = {}
    locked_path = Path(args.locked_file)
    if locked_path.exists():
        with open(locked_path, "r") as f:
            locked_data = json.load(f)
    else:
        print(f"Warning: Locked thresholds file {locked_path} not found. Using config.yaml defaults.")
        
    all_summaries = []
    for cat in args.categories:
        cat_locked = locked_data.get(cat, {})
        try:
            summary = evaluate_unbiased_category(
                category=cat,
                locked_params=cat_locked,
                dataset_root=args.dataset_root,
                device=device,
                out_dir=out_dir
            )
            all_summaries.append(summary)
        except Exception as e:
            print(f"Error evaluating category '{cat}': {e}", file=sys.stderr)
            
    if all_summaries:
        summary_csv = out_dir / "multi_category_summary.csv"
        headers = [
            "category", "image_auroc", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou",
            "detection_rate", "normal_accuracy", "locked_image_threshold", "locked_pixel_threshold",
            "use_spatial_prior", "total_test_images"
        ]
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({
                    "category": s["category"],
                    "image_auroc": f"{s['image_auroc']:.4f}",
                    "pixel_precision": f"{s['pixel_precision']:.4f}",
                    "pixel_recall": f"{s['pixel_recall']:.4f}",
                    "pixel_f1": f"{s['pixel_f1']:.4f}",
                    "pixel_iou": f"{s['pixel_iou']:.4f}",
                    "detection_rate": f"{s['detection_rate']*100:.1f}%",
                    "normal_accuracy": f"{s['normal_accuracy']*100:.1f}%",
                    "locked_image_threshold": f"{s['locked_image_threshold']:.2f}",
                    "locked_pixel_threshold": f"{s['locked_pixel_threshold']:.2f}",
                    "use_spatial_prior": s["use_spatial_prior"],
                    "total_test_images": s["total_test_images"]
                })
        print(f"\nSaved final unbiased benchmark to: {summary_csv.resolve()}")

if __name__ == "__main__":
    main()
