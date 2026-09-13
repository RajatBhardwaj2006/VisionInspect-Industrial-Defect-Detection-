"""
Phase 3.2 - Final Unbiased Evaluation Script
=============================================
Runs the final MVTec test evaluation using Phase 3.2 locked parameters.
Test set is evaluated ONCE after parameters are locked.
No threshold tuning on test data.

Compares Phase 3.1 vs Phase 3.2 results side by side.

Usage:
    python scripts/run_phase3_2_eval.py --categories bottle leather transistor zipper screw
"""
import os
import sys
import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.data.dataset_loader import MVTecTestDataset
from src.utils.config import load_config, get_category_config


def apply_localization_v32(
    smooth_map: np.ndarray,
    pixel_threshold: float,
    morph_kernel: int,
    min_area: int,
    border_margin: int
) -> Tuple[np.ndarray, List[Dict]]:
    """Phase 3.2 localization pipeline."""
    bin_mask = (smooth_map > pixel_threshold).astype(np.uint8)

    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    from scipy.ndimage import label as scipy_label
    labeled, num = scipy_label(opened)
    clean_mask = np.zeros_like(opened, dtype=bool)
    H, W = smooth_map.shape
    boxes = []

    for i in range(1, num + 1):
        comp = (labeled == i)
        if comp.sum() < min_area:
            continue
        ys, xs = np.where(comp)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        if (x_min < border_margin or x_max > W - 1 - border_margin or
                y_min < border_margin or y_max > H - 1 - border_margin):
            continue
        clean_mask[comp] = True
        boxes.append({"x": x_min, "y": y_min, "w": x_max - x_min + 1, "h": y_max - y_min + 1})

    return clean_mask, boxes


def evaluate_category_v32(
    category: str,
    v32_params: Dict[str, Any],
    dataset_root: str,
    device: torch.device,
    out_dir: Path
) -> Dict[str, Any]:
    """
    Evaluates final MVTec test set using Phase 3.2 locked parameters.
    Zero test-set calibration.
    """
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Phase 3.2 params (locked from calibration)
    img_th = v32_params.get("locked_image_threshold")
    pixel_th = v32_params.get("locked_pixel_threshold")
    gaussian_sigma = v32_params.get("locked_gaussian_sigma", 1.2)
    use_prior = v32_params.get("use_spatial_prior", True)
    prior_mode = v32_params.get("prior_mode", "mean")
    border_margin = v32_params.get("border_margin", 5)
    min_area = v32_params.get("min_region_area", 25)
    morph_kernel = v32_params.get("morphology_kernel", 3)
    merge_distance = v32_params.get("merge_distance", 15.0)

    print(f"\n{'='*65}")
    print(f"   PHASE 3.2 TEST EVALUATION: {category.upper()}")
    print(f"{'='*65}")
    print(f"  img_th={img_th:.3f}  pixel_th={pixel_th:.3f}  sigma={gaussian_sigma}")
    print(f"  prior={use_prior} ({prior_mode})  border_margin={border_margin}")
    print(f"  min_area={min_area}  morph_kernel={morph_kernel}  merge_dist={merge_distance}")

    model = PatchCoreModelV22(device=device)
    model.load(str(model_dir))

    # Load p75 prior if available
    p75_path = model_dir / "spatial_prior_p75.pt"
    if p75_path.exists() and model.normal_spatial_prior_p75 is None:
        model.normal_spatial_prior_p75 = torch.load(p75_path, map_location=device, weights_only=True).to(device)

    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category=category)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"  Test images: {len(test_dataset)}")

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
            score, amap_tensor = model.predict(img, use_prior=use_prior, prior_mode=prior_mode)
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

            # Apply Phase 3.2 localization
            clean_mask, _ = apply_localization_v32(
                smooth_map, pixel_th, morph_kernel, min_area, border_margin
            )

            is_pred_defective = (score > img_th) and (clean_mask.sum() > 0)
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

    try:
        image_auroc = float(roc_auc_score(image_labels, image_scores))
    except Exception:
        image_auroc = 0.50

    prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else 0.0

    total_normal = len(normal_scores)
    total_defective = len(defective_scores)
    total_samples = len(image_scores)
    detected_defective = sum(cat_metrics[d]['detected'] for d in cat_metrics if d != 'good')
    detection_rate = detected_defective / total_defective if total_defective > 0 else 0.0
    normal_accuracy = (total_normal - normal_fp_count) / total_normal if total_normal > 0 else 1.0

    print(f"\n  Results:")
    print(f"    AUROC:             {image_auroc:.4f}")
    print(f"    Pixel Precision:   {prec:.4f}")
    print(f"    Pixel Recall:      {rec:.4f}")
    print(f"    Pixel F1:          {f1:.4f}")
    print(f"    Pixel IoU:         {iou:.4f}")
    print(f"    Detection Rate:    {detected_defective}/{total_defective} ({detection_rate*100:.1f}%)")
    print(f"    Normal Accuracy:   {total_normal - normal_fp_count}/{total_normal} ({normal_accuracy*100:.1f}%)")

    # Save results
    cat_out = out_dir / category
    cat_out.mkdir(parents=True, exist_ok=True)

    results_summary = {
        "category": category,
        "phase": "3.2",
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
        "border_margin": border_margin,
        "min_region_area": min_area,
        "morphology_kernel": morph_kernel,
        "merge_distance": merge_distance,
        "prior_mode": prior_mode,
        "total_test_images": total_samples,
        "normal_images": total_normal,
        "defective_images": total_defective,
        "detected_defective": detected_defective,
        "normal_false_positives": normal_fp_count,
        "use_spatial_prior": use_prior,
        "tp_pixels": global_tp,
        "fp_pixels": global_fp,
        "fn_pixels": global_fn,
    }

    with open(cat_out / "metrics.json", "w") as f:
        json.dump(results_summary, f, indent=4)

    with open(cat_out / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in results_summary.items():
            writer.writerow([k, v])

    with open(cat_out / "per_defect.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defect_type", "images", "detected", "detection_rate",
                         "tp", "fp", "fn", "precision", "recall", "f1", "iou"])
        for dtype, m in sorted(cat_metrics.items()):
            imgs = m['imgs']
            det = m['detected']
            det_r = det / imgs if imgs > 0 else 0
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            iu = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            writer.writerow([dtype, imgs, det, f"{det_r:.4f}",
                             tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}", f"{iu:.4f}"])

    return results_summary


def load_phase31_results() -> Dict[str, Dict]:
    """Load Phase 3.1 results for comparison."""
    results = {}
    phase31_dir = Path("results/phase3/final_test")
    for cat_dir in phase31_dir.iterdir():
        if cat_dir.is_dir():
            metrics_path = cat_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    results[cat_dir.name] = json.load(f)
    return results


def generate_comparison_report(
    phase31: Dict[str, Dict],
    phase32: Dict[str, Dict],
    out_path: Path
):
    """Generate Phase 3.1 vs Phase 3.2 side-by-side comparison CSV."""
    metrics = ["image_auroc", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou",
               "detection_rate", "normal_accuracy"]
    metric_labels = ["AUROC", "Precision", "Recall", "F1", "IoU", "Det.Rate", "Normal Acc"]

    rows = []
    for category in sorted(set(list(phase31.keys()) + list(phase32.keys()))):
        p31 = phase31.get(category, {})
        p32 = phase32.get(category, {})
        for m, label in zip(metrics, metric_labels):
            v31 = p31.get(m, None)
            v32 = p32.get(m, None)
            delta = None
            if v31 is not None and v32 is not None:
                delta = round(float(v32) - float(v31), 4)
                direction = "+" if delta > 0 else ("-" if delta < 0 else "=")
            else:
                direction = "?"
            rows.append({
                "category": category,
                "metric": label,
                "phase_3_1": round(float(v31), 4) if v31 is not None else "N/A",
                "phase_3_2": round(float(v32), 4) if v32 is not None else "N/A",
                "delta": delta if delta is not None else "N/A",
                "direction": direction
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "metric", "phase_3_1", "phase_3_2", "delta", "direction"])
        writer.writeheader()
        writer.writerows(rows)

    # Compute and save macro averages
    macro_path = out_path.parent / "macro_averages.json"
    macro = {}
    for phase, data in [("phase_3_1", phase31), ("phase_3_2", phase32)]:
        macro[phase] = {}
        for m in metrics:
            vals = [float(data[cat].get(m, 0.0)) for cat in data if m in data[cat]]
            if vals:
                macro[phase][m] = round(float(np.mean(vals)), 4)

    # Compute deltas
    macro["delta"] = {}
    for m in metrics:
        v31 = macro.get("phase_3_1", {}).get(m)
        v32 = macro.get("phase_3_2", {}).get(m)
        if v31 is not None and v32 is not None:
            macro["delta"][m] = round(v32 - v31, 4)

    with open(macro_path, "w", encoding="utf-8") as f:
        json.dump(macro, f, indent=4)

    return rows, macro


def print_comparison_table(phase31, phase32, macro):
    """Print formatted comparison table to stdout."""
    cats = sorted(set(list(phase31.keys()) + list(phase32.keys())))
    metrics = ["image_auroc", "pixel_precision", "pixel_recall", "pixel_f1",
               "pixel_iou", "detection_rate", "normal_accuracy"]
    labels = ["AUROC", "Prec", "Recall", "F1", "IoU", "Det%", "Norm%"]

    print(f"\n{'='*85}")
    print(f"  PHASE 3.1 vs PHASE 3.2 COMPARISON")
    print(f"{'='*85}")
    header = f"  {'Category':12s}"
    for lbl in labels:
        header += f"  {'P3.1':>6s} {'P3.2':>6s} {'diff':>5s}"
    print(header)
    print(f"  {'-'*82}")
    for cat in cats:
        p31 = phase31.get(cat, {})
        p32 = phase32.get(cat, {})
        row = f"  {cat:12s}"
        for m in metrics:
            v31 = float(p31.get(m, 0.0))
            v32 = float(p32.get(m, 0.0))
            delta = v32 - v31
            sym = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
            row += f"  {v31:6.4f} {v32:6.4f} {sym}{abs(delta):.3f}"
        print(row)

    # Macro averages
    print(f"  {'-'*82}")
    row = f"  {'MACRO AVG':12s}"
    for m in metrics:
        v31 = macro.get("phase_3_1", {}).get(m, 0.0)
        v32 = macro.get("phase_3_2", {}).get(m, 0.0)
        delta = macro.get("delta", {}).get(m, 0.0)
        sym = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
        row += f"  {v31:6.4f} {v32:6.4f} {sym}{abs(delta):.3f}"
    print(row)
    print(f"{'='*85}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3.2 Unbiased Evaluation")
    parser.add_argument("--categories", nargs="+",
                        default=["bottle", "leather", "transistor", "zipper", "screw"])
    parser.add_argument("--dataset-root", type=str, default="dataset/mvtec_anomaly_detection")
    parser.add_argument("--params-file", type=str,
                        default="results/phase3/phase3_2_localization/calibration/locked_params_v32.json")
    parser.add_argument("--output-dir", type=str,
                        default="results/phase3/phase3_2_localization/final_test")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only generate comparison report from already existing evaluations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 3.2 locked params
    params_path = Path(args.params_file)
    if not params_path.exists():
        raise FileNotFoundError(
            f"Phase 3.2 locked parameters not found: {params_path}\n"
            f"Run calibrate_phase3_2.py first."
        )
    with open(params_path) as f:
        all_v32_params = json.load(f)

    all_summaries_v32 = {}

    if args.compare_only:
        for cat in args.categories:
            cat_m = out_dir / cat / "metrics.json"
            if cat_m.exists():
                with open(cat_m) as f:
                    all_summaries_v32[cat] = json.load(f)
    else:
        for category in args.categories:
            v32_params = all_v32_params.get(category, {})
            if not v32_params:
                print(f"[SKIP] No Phase 3.2 params found for '{category}'")
                continue
            try:
                summary = evaluate_category_v32(
                    category=category,
                    v32_params=v32_params,
                    dataset_root=args.dataset_root,
                    device=device,
                    out_dir=out_dir
                )
                all_summaries_v32[category] = summary
            except Exception as e:
                print(f"ERROR evaluating '{category}': {e}")
                import traceback
                traceback.print_exc()

    # Save multi-category summary CSV
    summary_csv = out_dir / "multi_category_summary.csv"
    if all_summaries_v32:
        headers = ["category", "image_auroc", "pixel_precision", "pixel_recall",
                   "pixel_f1", "pixel_iou", "detection_rate", "normal_accuracy",
                   "locked_image_threshold", "locked_pixel_threshold",
                   "border_margin", "min_region_area", "morphology_kernel",
                   "prior_mode", "total_test_images"]
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for cat, s in all_summaries_v32.items():
                writer.writerow(s)
        print(f"\n  Phase 3.2 summary saved: {summary_csv}")

    # Load Phase 3.1 results and generate comparison
    phase31 = load_phase31_results()
    if phase31 and all_summaries_v32:
        comp_path = out_dir / "phase31_vs_phase32_comparison.csv"
        _, macro = generate_comparison_report(phase31, all_summaries_v32, comp_path)
        print(f"  Comparison report saved: {comp_path}")
        print_comparison_table(phase31, all_summaries_v32, macro)

        # Save macro averages
        macro_out = out_dir / "macro_averages.json"
        with open(macro_out, "w") as f:
            json.dump(macro, f, indent=4)
        print(f"  Macro averages saved: {macro_out}")


if __name__ == "__main__":
    main()
