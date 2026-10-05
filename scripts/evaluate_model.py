"""
Industrial Anomaly Detection - Comprehensive Model Evaluation Script
===================================================================
Evaluates PatchCore detector on the MVTec AD test set for a given category.
Provides:
  - Confusion Matrix (TP, FP, TN, FN)
  - Classification Metrics (Accuracy, Precision, Recall, Specificity, F1-score)
  - Curve Metrics (ROC-AUC, PR-AUC / Average Precision)
  - Defect-Specific Breakdown (Recall and scores per defect type)
  - Defect Localization Metrics (Pixel IoU, Region GT overlap rate)
  - Failure Case Analysis (False Positives and False Negatives)
  - Target Sample Deep-Dive (e.g., thread_side/005.png)

Usage:
    python scripts/evaluate_model.py --category screw
    python scripts/evaluate_model.py --category bottle
    python scripts/evaluate_model.py --category zipper
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

from src.detection.patchcore_v23_detector import PatchCoreDetectorV23


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def evaluate_category(category: str, data_dir: str, target_stem: str = "005", target_defect: str = "thread_side") -> Dict[str, Any]:
    dataset_root = Path(data_dir) / category
    test_dir = dataset_root / "test"
    gt_dir = dataset_root / "ground_truth"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    print("=" * 80)
    print(f"  VISIONINSPECT MODEL EVALUATION: {category.upper()}")
    print("=" * 80)

    # Initialize Detector
    detector = PatchCoreDetectorV23(category=category)

    samples = []
    for defect_folder in sorted(test_dir.iterdir()):
        if not defect_folder.is_dir():
            continue
        defect_type = defect_folder.name
        for img_path in sorted(defect_folder.glob("*.png")):
            samples.append((img_path, defect_type))

    print(f"\nEvaluating {len(samples)} test samples for category '{category}'...\n")

    y_true = []
    y_pred = []
    y_scores = []

    per_defect_stats = defaultdict(lambda: {"total": 0, "detected": 0, "scores": [], "ious": []})
    sample_records = []
    target_sample_result = None

    for idx, (img_path, defect_type) in enumerate(samples):
        is_defective = (defect_type != "good")
        stem = img_path.stem

        # Run inference
        result = detector.inspect(img_path)
        pred_defective = (result["status"] == "DEFECTIVE")
        score = float(result["score"])
        margin = float(result["decision_margin"])

        y_true.append(1 if is_defective else 0)
        y_pred.append(1 if pred_defective else 0)
        y_scores.append(score)

        iou = 0.0
        gt_overlap_pixels = 0
        gt_area = 0

        # Evaluate localization if ground truth mask exists
        if is_defective:
            gt_mask_path = gt_dir / defect_type / f"{stem}_mask.png"
            if gt_mask_path.exists():
                gt_img = Image.open(gt_mask_path).convert("L")
                gt_np = np.array(gt_img.resize((256, 256), resample=Image.NEAREST)) > 127
                gt_area = int(gt_np.sum())

                # Binary prediction mask from detector (256x256)
                pred_mask = result.get("clean_mask", np.zeros((256, 256), dtype=bool))
                iou = compute_mask_iou(pred_mask, gt_np)
                gt_overlap_pixels = int(np.logical_and(pred_mask, gt_np).sum())
                per_defect_stats[defect_type]["ious"].append(iou)

        per_defect_stats[defect_type]["total"] += 1
        if pred_defective if is_defective else not pred_defective:
            per_defect_stats[defect_type]["detected"] += 1
        per_defect_stats[defect_type]["scores"].append(score)

        record = {
            "path": str(img_path),
            "defect_type": defect_type,
            "stem": stem,
            "ground_truth": "DEFECTIVE" if is_defective else "NORMAL",
            "prediction": result["status"],
            "score": round(score, 4),
            "threshold": round(result["image_threshold"], 4),
            "margin": round(margin, 4),
            "regions_count": len(result["localized_regions"]),
            "iou": round(iou, 4) if is_defective else None,
            "gt_overlap": gt_overlap_pixels if is_defective else None,
            "is_correct": (is_defective == pred_defective)
        }
        sample_records.append(record)

        if defect_type == target_defect and stem == target_stem:
            target_sample_result = {
                **record,
                "regions": result["localized_regions"]
            }

    # Aggregate Metrics Calculation
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()

    accuracy = float((tp + tn) / len(y_true_arr))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
    except Exception:
        pr_auc = 0.0

    all_ious = [r["iou"] for r in sample_records if r["iou"] is not None]
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    defect_hit_rate = float(np.mean([iou > 0.05 for iou in all_ious])) if all_ious else 0.0

    # Print Summary Report
    print("=" * 80)
    print("  CONFUSION MATRIX")
    print("=" * 80)
    print(f"  True Positives  (TP) : {tp:3d}  (Defective correctly classified)")
    print(f"  True Negatives  (TN) : {tn:3d}  (Normal correctly classified)")
    print(f"  False Positives (FP) : {fp:3d}  (Normal falsely classified as Defective)")
    print(f"  False Negatives (FN) : {fn:3d}  (Defective missed as Normal)")
    print(f"  Total Samples        : {len(y_true_arr):3d}")

    print("\n" + "=" * 80)
    print("  CLASSIFICATION & DETECTION METRICS")
    print("=" * 80)
    print(f"  Accuracy             : {accuracy * 100:6.2f}%")
    print(f"  Precision            : {precision * 100:6.2f}%")
    print(f"  Recall (Sensitivity) : {recall * 100:6.2f}%")
    print(f"  Specificity          : {specificity * 100:6.2f}%")
    print(f"  F1-Score             : {f1:8.4f}")
    print(f"  ROC-AUC              : {roc_auc * 100:6.2f}%")
    print(f"  PR-AUC (Avg Prec)    : {pr_auc * 100:6.2f}%")
    print(f"  Mean Pixel IoU (Def) : {mean_iou * 100:6.2f}%")
    print(f"  Defect Region Hit Rt : {defect_hit_rate * 100:6.2f}%")

    print("\n" + "=" * 80)
    print("  PER-DEFECT TYPE BREAKDOWN")
    print("=" * 80)
    print(f"  {'Defect Type':<22} | {'Count':<5} | {'Detected':<8} | {'Recall/Acc':<10} | {'Mean Score':<10} | {'Score Range'}")
    print("  " + "-" * 76)
    for def_name, st in sorted(per_defect_stats.items()):
        cnt = st["total"]
        det = st["detected"]
        rate = (det / cnt * 100.0) if cnt > 0 else 0.0
        m_score = np.mean(st["scores"]) if st["scores"] else 0.0
        min_s = min(st["scores"]) if st["scores"] else 0.0
        max_s = max(st["scores"]) if st["scores"] else 0.0
        print(f"  {def_name:<22} | {cnt:<5} | {det:<8} | {rate:6.1f}%    | {m_score:10.4f} | [{min_s:.4f}, {max_s:.4f}]")

    # Target Sample Inspection
    if target_sample_result:
        print("\n" + "=" * 80)
        print(f"  TARGET SAMPLE DEEP-DIVE: {target_defect}/{target_stem}.png")
        print("=" * 80)
        print(f"  Path                 : {target_sample_result['path']}")
        print(f"  Ground Truth         : {target_sample_result['ground_truth']}")
        print(f"  Inspection Status    : {target_sample_result['prediction']} ({'SUCCESS' if target_sample_result['is_correct'] else 'FAILURE'})")
        print(f"  Anomaly Score        : {target_sample_result['score']:.4f}")
        print(f"  Calibrated Threshold : {target_sample_result['threshold']:.4f}")
        print(f"  Decision Margin      : {target_sample_result['margin']:+.4f}")
        print(f"  Localized Regions    : {target_sample_result['regions_count']}")
        print(f"  Pixel IoU with GT    : {target_sample_result['iou']:.4f}")
        print(f"  Overlap with GT Pix  : {target_sample_result['gt_overlap']} px")
        if target_sample_result.get("regions"):
            for ridx, reg in enumerate(target_sample_result["regions"]):
                print(f"    Region {ridx+1:02d}: bbox=[x={reg['x']}, y={reg['y']}, w={reg['width']}, h={reg['height']}], "
                      f"area={reg['area']}, max_anomaly={reg.get('max_val', 0.0):.4f}, intensity='{reg.get('intensity', 'N/A')}'")

    # Failure Cases
    fps = [r for r in sample_records if r["ground_truth"] == "NORMAL" and r["prediction"] == "DEFECTIVE"]
    fns = [r for r in sample_records if r["ground_truth"] == "DEFECTIVE" and r["prediction"] == "NORMAL"]

    print("\n" + "=" * 80)
    print(f"  FAILURE ANALYSIS: {len(fps)} False Positives, {len(fns)} False Negatives")
    print("=" * 80)
    if fps:
        print(f"  False Positives (Normal samples flagged as Defective, threshold={detector.image_threshold:.4f}):")
        for fp_item in fps[:10]:
            print(f"    - {Path(fp_item['path']).name} : score={fp_item['score']:.4f}, margin={fp_item['margin']:+.4f}")
        if len(fps) > 10:
            print(f"    ... and {len(fps) - 10} more.")
    else:
        print("  Zero False Positives! All normal samples correctly verified.")

    if fns:
        print(f"\n  False Negatives (Defective samples missed as Normal, threshold={detector.image_threshold:.4f}):")
        for fn_item in fns[:10]:
            print(f"    - {fn_item['defect_type']}/{Path(fn_item['path']).name} : score={fn_item['score']:.4f}, margin={fn_item['margin']:+.4f}")
        if len(fns) > 10:
            print(f"    ... and {len(fns) - 10} more.")
    else:
        print("  Zero False Negatives! 100% defect recall achieved.")

    # Save output report
    out_dir = Path("results/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"evaluation_{category}.json"
    summary_report = {
        "category": category,
        "metrics": {
            "total_samples": len(y_true_arr),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "mean_iou": round(mean_iou, 4),
            "defect_hit_rate": round(defect_hit_rate, 4)
        },
        "config": {
            "model_dir": detector.model_dir,
            "use_spatial_prior": detector.use_spatial_prior,
            "score_method": detector.image_score_method,
            "image_threshold": detector.image_threshold,
            "pixel_threshold": detector.pixel_threshold,
            "border_margin": detector.border_margin,
            "min_region_area": detector.min_region_area
        },
        "target_sample": target_sample_result,
        "per_defect_stats": {
            k: {
                "total": v["total"],
                "detected": v["detected"],
                "recall": round(v["detected"] / v["total"], 4) if v["total"] > 0 else 0.0,
                "mean_score": round(float(np.mean(v["scores"])), 4) if v["scores"] else 0.0
            } for k, v in per_defect_stats.items()
        }
    }
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\nEvaluation report successfully saved to: {report_path}")
    print("=" * 80 + "\n")
    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VisionInspect Anomaly Detection Model")
    parser.add_argument("--category", type=str, default="screw", help="MVTec category to evaluate (default: screw)")
    parser.add_argument("--data_dir", type=str, default="dataset/mvtec_anomaly_detection", help="Dataset root path")
    parser.add_argument("--target_stem", type=str, default="005", help="Target test sample stem for deep-dive")
    parser.add_argument("--target_defect", type=str, default="thread_side", help="Target defect folder for deep-dive")
    args = parser.parse_args()

    evaluate_category(category=args.category, data_dir=args.data_dir, target_stem=args.target_stem, target_defect=args.target_defect)
