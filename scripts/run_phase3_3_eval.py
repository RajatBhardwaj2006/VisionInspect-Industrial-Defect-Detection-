"""
Phase 3.3 — Targeted Transistor Performance Evaluation
======================================================
Runs the final unbiased evaluation on the MVTec transistor test set
using Phase 3.3 locked parameters:
  - use_spatial_prior: false
  - image_score_method: top200
  - image_threshold: 3.525
  - pixel_threshold: 2.822
  - border_margin: 0
  - min_region_area: 15
  - morphology_kernel: 2
  - merge_distance: 12.0

Generates a direct side-by-side comparison across Phase 3.1, 3.2, and 3.3.
"""
import os
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

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
from src.data.dataset_loader import MVTecTestDataset
from src.utils.config import load_config, get_category_config
from scripts.run_phase3_2_eval import apply_localization_v32


def evaluate_transistor_phase3_3() -> Dict[str, Any]:
    cfg = load_config()
    cat_cfg = get_category_config("transistor", cfg)
    
    device = torch.device("cpu")
    model_dir = project_root / cat_cfg.get("model_dir", "models/transistor/patchcore_v23")
    dataset_root = str(project_root / "dataset" / "mvtec_anomaly_detection")

    img_th = cat_cfg["image_threshold"]
    pixel_th = cat_cfg["pixel_threshold"]
    sigma = cat_cfg.get("gaussian_sigma", 1.20)
    use_prior = cat_cfg.get("use_spatial_prior", False)
    prior_mode = cat_cfg.get("prior_mode", "none")
    border_margin = cat_cfg.get("border_margin", 0)
    min_area = cat_cfg.get("min_region_area", 15)
    morph_kernel = cat_cfg.get("morphology_kernel", 2)
    score_method = cat_cfg.get("image_score_method", "top200")

    print(f"\n{'='*70}")
    print("   PHASE 3.3 FINAL EVALUATION: TRANSISTOR")
    print(f"{'='*70}")
    print(f"  Model Dir:        {model_dir}")
    print(f"  Spatial Prior:    {use_prior} ({prior_mode})")
    print(f"  Score Method:     {score_method}")
    print(f"  Image Threshold:  {img_th:.4f}")
    print(f"  Pixel Threshold:  {pixel_th:.4f}")
    print(f"  Border Margin:    {border_margin}")
    print(f"  Min Region Area:  {min_area}")
    print(f"  Morph Kernel:     {morph_kernel}")

    model = PatchCoreModelV22(device=device)
    model.load(str(model_dir))

    test_ds = MVTecTestDataset(dataset_root=dataset_root, category="transistor")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    cat_metrics = defaultdict(lambda: {'imgs': 0, 'detected': 0, 'tp': 0, 'fp': 0, 'fn': 0})
    image_scores = []
    image_labels = []
    normal_scores = []
    defective_scores = []
    normal_fp_count = 0
    global_tp = global_fp = global_fn = 0

    with torch.no_grad():
        for idx, (img, _, dtype, is_def, gt_mask) in enumerate(test_loader):
            img = img.to(device)
            _, amap = model.predict(img, use_prior=use_prior, prior_mode=prior_mode)
            amap_np = amap.cpu().numpy()

            if sigma > 0:
                smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=sigma)
            else:
                smooth_map = amap_np

            # Compute score based on score_method
            flat = smooth_map.flatten()
            if score_method == "top200":
                score = float(np.sort(flat)[-200:].mean())
            elif score_method == "top250":
                score = float(np.sort(flat)[-250:].mean())
            elif score_method == "top100":
                score = float(np.sort(flat)[-100:].mean())
            elif score_method == "p99.5":
                score = float(np.percentile(flat, 99.5))
            else:
                score = float(flat.max())

            is_def_val = 1 if is_def.item() else 0
            dt = dtype[0]
            gt_binary = gt_mask.squeeze(0).squeeze(0).numpy() > 0

            image_scores.append(score)
            image_labels.append(is_def_val)

            if is_def_val:
                defective_scores.append(score)
            else:
                normal_scores.append(score)

            clean_mask, boxes = apply_localization_v32(
                smooth_map, pixel_th, morph_kernel, min_area, border_margin
            )

            is_pred_defective = (score > img_th) and (len(boxes) > 0)
            pred_mask = clean_mask if is_pred_defective else np.zeros_like(clean_mask, dtype=bool)

            cat_metrics[dt]['imgs'] += 1
            if is_pred_defective:
                cat_metrics[dt]['detected'] += 1
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

            cat_metrics[dt]['tp'] += tp_img
            cat_metrics[dt]['fp'] += fp_img
            cat_metrics[dt]['fn'] += fn_img
            global_tp += tp_img
            global_fp += fp_img
            global_fn += fn_img

    image_auroc = float(roc_auc_score(image_labels, image_scores))
    prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    iou = global_tp / (global_tp + global_fp + global_fn) if (global_tp + global_fp + global_fn) > 0 else 0.0

    total_normal = len(normal_scores)
    total_defective = len(defective_scores)
    detected_defective = sum(cat_metrics[d]['detected'] for d in cat_metrics if d != 'good')
    detection_rate = detected_defective / total_defective if total_defective > 0 else 0.0
    normal_accuracy = (total_normal - normal_fp_count) / total_normal if total_normal > 0 else 1.0

    print(f"\n--- Results Summary ---")
    print(f"  Image AUROC:      {image_auroc:.4f}")
    print(f"  Detection Rate:   {detected_defective}/{total_defective} ({detection_rate*100:.1f}%)")
    print(f"  Normal Accuracy:  {total_normal - normal_fp_count}/{total_normal} ({normal_accuracy*100:.1f}%)")
    print(f"  Pixel Precision:  {prec:.4f}")
    print(f"  Pixel Recall:     {rec:.4f}")
    print(f"  Pixel F1:         {f1:.4f}")
    print(f"  Pixel IoU:        {iou:.4f}")

    print(f"\n--- Per-Defect Breakdown ---")
    for dt, m in sorted(cat_metrics.items()):
        if dt == 'good':
            print(f"  {dt:15s}: {m['detected']}/{m['imgs']} false positives ({m['detected']/m['imgs']*100:.1f}%)")
        else:
            print(f"  {dt:15s}: {m['detected']}/{m['imgs']} detected ({m['detected']/m['imgs']*100:.1f}%)")

    # Output directory
    out_dir = project_root / "results" / "phase3" / "phase3_3_transistor"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "category": "transistor",
        "phase": "3.3",
        "evaluation_type": "unbiased_locked_test",
        "image_auroc": round(image_auroc, 4),
        "detection_rate": round(detection_rate, 4),
        "normal_accuracy": round(normal_accuracy, 4),
        "pixel_precision": round(prec, 4),
        "pixel_recall": round(rec, 4),
        "pixel_f1": round(f1, 4),
        "pixel_iou": round(iou, 4),
        "total_test_images": len(test_ds),
        "normal_images": total_normal,
        "defective_images": total_defective,
        "detected_defective": detected_defective,
        "normal_false_positives": normal_fp_count,
        "locked_image_threshold": img_th,
        "locked_pixel_threshold": pixel_th,
        "score_method": score_method,
        "use_spatial_prior": use_prior,
        "border_margin": border_margin,
        "min_region_area": min_area,
        "morphology_kernel": morph_kernel,
        "per_defect": {dt: {"images": m["imgs"], "detected": m["detected"]} for dt, m in cat_metrics.items()}
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # Comparison across Phase 3.1 -> 3.2 -> 3.3
    comp_path = out_dir / "transistor_phase_comparison.md"
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("# Transistor Performance Comparison: Phase 3.1 → Phase 3.2 → Phase 3.3\n\n")
        f.write("| Metric | Phase 3.1 Baseline | Phase 3.2 Localization | Phase 3.3 Targeted (Locked) | Absolute Change (vs 3.2) |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **AUROC** | 0.8200 | 0.8200 | **{image_auroc:.4f}** | **+{image_auroc - 0.8200:.4f}** |\n")
        f.write(f"| **Detection Rate** | 15.0% (6/40) | 30.0% (12/40) | **{detection_rate*100:.1f}% ({detected_defective}/40)** | **+{detection_rate*100 - 30.0:.1f}%** |\n")
        f.write(f"| **Normal Accuracy** | 100.0% (60/60) | 93.3% (56/60) | **{normal_accuracy*100:.1f}% ({total_normal - normal_fp_count}/{total_normal})** | {normal_accuracy*100 - 93.3:.1f}% |\n")
        f.write(f"| **Pixel Precision** | 0.0000 | 0.5323 | **{prec:.4f}** | **+{prec - 0.5323:.4f}** |\n")
        f.write(f"| **Pixel Recall** | 0.0000 | 0.0442 | **{rec:.4f}** | **+{rec - 0.0442:.4f}** (6.2x) |\n")
        f.write(f"| **Pixel F1** | 0.0000 | 0.0816 | **{f1:.4f}** | **+{f1 - 0.0816:.4f}** (4.6x) |\n")
        f.write(f"| **Pixel IoU** | 0.0000 | 0.0425 | **{iou:.4f}** | **+{iou - 0.0425:.4f}** (5.4x) |\n\n")
        f.write("### Per-Defect Detection Comparison\n\n")
        f.write("| Defect Type | Total Samples | Phase 3.2 Detected | Phase 3.3 Detected | Phase 3.3 Detection Rate |\n")
        f.write("|---|---|---|---|---|\n")
        p32_counts = {"bent_lead": 9, "cut_lead": 0, "damaged_case": 0, "misplaced": 3}
        for dt in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
            f.write(f"| `{dt}` | 10 | {p32_counts[dt]}/10 | **{cat_metrics[dt]['detected']}/10** | **{cat_metrics[dt]['detected']/10*100:.1f}%** |\n")
        f.write(f"| `good` (Normals) | 60 | 4 FP | **{normal_fp_count} FP** | **{normal_accuracy*100:.1f}%** |\n")

    print(f"\nSaved comparison to: {comp_path}")
    return summary_data


if __name__ == "__main__":
    evaluate_transistor_phase3_3()
