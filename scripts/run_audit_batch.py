"""
VisionInspect - Real-World Batch Inspection & Final Audit Script
===============================================================
Runs real batch inference across 5 categories (bottle, leather, transistor, zipper, screw)
Target: 25 distinct images per category = 125 total real images.

Saves:
  - Original images in results/Updated_model_predictions/<category>/normal or defective
  - Visualizations (heatmap, localization, combined overlay) in results/Updated_model_predictions/<category>/visualizations
  - Structured metadata in results/Updated_model_predictions/audit_summary.json
  - Full audit report in results/Updated_model_predictions/AUDIT_REPORT.md
"""
import os
import sys
import json
import time
import shutil
import base64
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
from src.utils.config import load_config, get_category_config


OUTPUT_BASE = project_root / "results" / "Updated_model_predictions"
DATASET_BASE = project_root / "dataset" / "mvtec_anomaly_detection"

# Genuinely diverse, distinct selection of 25 real images per category
# Covering every available defect type and multiple normal samples
BATCH_SELECTION = {
    "bottle": {
        "good": ["000.png", "003.png", "006.png", "009.png", "012.png", "015.png", "018.png"],  # 7
        "broken_large": ["000.png", "003.png", "006.png", "009.png", "012.png", "015.png"],      # 6
        "broken_small": ["000.png", "003.png", "006.png", "009.png", "012.png", "015.png"],      # 6
        "contamination": ["000.png", "003.png", "006.png", "009.png", "012.png", "015.png"]      # 6
    },
    "leather": {
        "good": ["000.png", "005.png", "010.png", "015.png", "020.png"],                         # 5
        "cut": ["000.png", "004.png", "008.png", "012.png"],                                      # 4
        "color": ["000.png", "004.png", "008.png", "012.png"],                                    # 4
        "fold": ["000.png", "004.png", "008.png", "012.png"],                                     # 4
        "glue": ["000.png", "004.png", "008.png", "012.png"],                                     # 4
        "poke": ["000.png", "004.png", "008.png", "012.png"]                                      # 4
    },
    "transistor": {
        "good": ["000.png", "005.png", "010.png", "015.png", "020.png", "025.png", "030.png", "040.png", "050.png"], # 9
        "bent_lead": ["000.png", "002.png", "005.png", "008.png"],                                # 4
        "cut_lead": ["000.png", "002.png", "005.png", "008.png"],                                 # 4
        "damaged_case": ["000.png", "002.png", "005.png", "008.png"],                             # 4
        "misplaced": ["000.png", "002.png", "005.png", "008.png"]                                 # 4
    },
    "zipper": {
        "good": ["000.png", "005.png", "010.png", "015.png", "020.png"],                         # 5
        "broken_teeth": ["000.png", "005.png", "010.png"],                                        # 3
        "combined": ["000.png", "005.png", "010.png"],                                            # 3
        "fabric_border": ["000.png", "005.png", "010.png"],                                       # 3
        "fabric_interior": ["000.png", "005.png", "010.png"],                                     # 3
        "rough": ["000.png", "005.png", "010.png"],                                               # 3
        "split_teeth": ["000.png", "005.png", "010.png"],                                         # 3
        "squeezed_teeth": ["000.png", "005.png"]                                                   # 2
    },
    "screw": {
        "good": ["000.png", "005.png", "010.png", "020.png", "030.png"],                         # 5
        "thread_side": ["001.png", "002.png", "005.png", "007.png", "010.png", "014.png"],        # 6 (includes 005)
        "thread_top": ["000.png", "003.png", "008.png", "012.png"],                               # 4
        "scratch_head": ["000.png", "004.png", "010.png"],                                        # 3
        "scratch_neck": ["000.png", "002.png", "008.png", "015.png"],                             # 4
        "manipulated_front": ["000.png", "005.png", "012.png"]                                    # 3
    }
}


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, int, int]:
    """Compute IoU, intersection pixels, and union pixels."""
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    iou = float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)
    return iou, inter, union


def run_batch_audit(device: torch.device) -> Dict[str, Any]:
    print("=" * 80)
    print("  VISIONINSPECT — REAL-WORLD BATCH INSPECTION & AUDIT")
    print(f"  Device: {device}")
    print(f"  Output Directory: {OUTPUT_BASE}")
    print("=" * 80)

    # Initialize directory structure
    categories = ["bottle", "leather", "transistor", "zipper", "screw"]
    for cat in categories:
        (OUTPUT_BASE / cat / "normal").mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE / cat / "defective").mkdir(parents=True, exist_ok=True)
        (OUTPUT_BASE / cat / "visualizations").mkdir(parents=True, exist_ok=True)

    all_records = []
    category_summaries = {}

    for cat in categories:
        print(f"\n[{cat.upper()}] Initializing detector...")
        detector = PatchCoreDetectorV23(category=cat, device=device)
        cat_dir = DATASET_BASE / cat
        gt_dir = cat_dir / "ground_truth"

        selection = BATCH_SELECTION[cat]
        cat_records = []
        latencies = []

        for defect_type, filenames in selection.items():
            for fname in filenames:
                img_path = cat_dir / "test" / defect_type / fname
                if not img_path.exists():
                    print(f"  [WARN] File not found: {img_path}")
                    continue

                stem = img_path.stem
                is_defective = (defect_type != "good")

                t0 = time.perf_counter()
                result = detector.inspect(str(img_path))
                latency_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(latency_ms)

                pred_status = result["status"]
                is_pred_defective = (pred_status == "DEFECTIVE")
                is_correct = (is_defective == is_pred_defective)

                # Ground truth mask & localization analysis
                iou = None
                gt_overlap_px = None
                gt_area_px = None
                if is_defective:
                    gt_mask_path = gt_dir / defect_type / f"{stem}_mask.png"
                    if gt_mask_path.exists():
                        gt_pil = Image.open(gt_mask_path).convert("L")
                        gt_arr = np.array(gt_pil.resize((256, 256), resample=Image.NEAREST)) > 127
                        gt_area_px = int(gt_arr.sum())

                        pred_mask = result.get("clean_mask", np.zeros((256, 256), dtype=bool))
                        iou, gt_overlap_px, _ = compute_mask_iou(pred_mask, gt_arr)

                # Deterministic filename: {category}_{defect_type}_{stem}_{prediction}_{version}.png
                pred_tag = "DEFECTIVE" if is_pred_defective else "NORMAL"
                base_name = f"{cat}_{defect_type}_{stem}_{pred_tag}_v23"

                # 1. Save original image copy into normal/ or defective/ based on prediction
                dest_subfolder = "defective" if is_pred_defective else "normal"
                orig_dest = OUTPUT_BASE / cat / dest_subfolder / f"{base_name}_orig.png"
                shutil.copyfile(img_path, orig_dest)

                # 2. Decode and save Heatmap
                hm_bytes = base64.b64decode(result["heatmap_base64"])
                hm_dest = OUTPUT_BASE / cat / "visualizations" / f"{base_name}_heatmap.png"
                with open(hm_dest, "wb") as f:
                    f.write(hm_bytes)

                # 3. Decode and save Localization overlay
                ov_bytes = base64.b64decode(result["overlay_base64"])
                ov_dest = OUTPUT_BASE / cat / "visualizations" / f"{base_name}_localization.png"
                with open(ov_dest, "wb") as f:
                    f.write(ov_bytes)

                # 4. Save combined side-by-side visualization
                orig_img = cv2.imread(str(img_path))
                hm_img = cv2.imread(str(hm_dest))
                ov_img = cv2.imread(str(ov_dest))

                # Resize to standard height for side-by-side
                std_h = 320
                def resize_to_h(im, h):
                    scale = h / float(im.shape[0])
                    return cv2.resize(im, (int(im.shape[1] * scale), h))

                combined_row = np.hstack([
                    resize_to_h(orig_img, std_h),
                    resize_to_h(hm_img, std_h),
                    resize_to_h(ov_img, std_h)
                ])
                comb_dest = OUTPUT_BASE / cat / "visualizations" / f"{base_name}_combined.png"
                cv2.imwrite(str(comb_dest), combined_row)

                max_anomaly_val = float(result.get("score", 0.0))
                if result.get("localized_regions"):
                    max_anomaly_val = max([r.get("max_val", 0.0) for r in result["localized_regions"]] + [max_anomaly_val])

                record = {
                    "category": cat,
                    "filename": f"{stem}.png",
                    "original_path": str(img_path),
                    "defect_type": defect_type,
                    "ground_truth": "defective" if is_defective else "normal",
                    "prediction": "defective" if is_pred_defective else "normal",
                    "is_correct": is_correct,
                    "anomaly_score": round(float(result["score"]), 4),
                    "image_threshold": round(float(result["image_threshold"]), 4),
                    "decision_margin": round(float(result["decision_margin"]), 4),
                    "pixel_threshold": round(float(result["pixel_threshold"]), 4),
                    "localized_region_count": len(result["localized_regions"]),
                    "bounding_boxes": [
                        {
                            "x": r["x"], "y": r["y"], "width": r["width"], "height": r["height"],
                            "area": r["area"], "max_val": round(r.get("max_val", 0.0), 4),
                            "intensity": r.get("intensity", "")
                        } for r in result["localized_regions"]
                    ],
                    "max_anomaly_value": round(max_anomaly_val, 4),
                    "iou_with_gt": round(iou, 4) if iou is not None else None,
                    "gt_overlap_pixels": gt_overlap_px,
                    "gt_area_pixels": gt_area_px,
                    "inference_latency_ms": round(latency_ms, 2),
                    "model_version": f"patchcore_v{result['model_version']}",
                    "scoring_method": detector.image_score_method,
                    "use_spatial_prior": detector.use_spatial_prior,
                    "prior_mode": detector.prior_mode,
                    "saved_visualizations": {
                        "original": str(orig_dest.relative_to(OUTPUT_BASE)),
                        "heatmap": str(hm_dest.relative_to(OUTPUT_BASE)),
                        "localization": str(ov_dest.relative_to(OUTPUT_BASE)),
                        "combined": str(comb_dest.relative_to(OUTPUT_BASE))
                    },
                    "timestamp": datetime.now().isoformat()
                }
                cat_records.append(record)
                all_records.append(record)

        # Calculate category summary metrics
        y_true = [1 if r["ground_truth"] == "defective" else 0 for r in cat_records]
        y_pred = [1 if r["prediction"] == "defective" else 0 for r in cat_records]
        y_scores = [r["anomaly_score"] for r in cat_records]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        acc = float((tp + tn) / len(y_true))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        try:
            auc = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc = 0.0

        def_ious = [r["iou_with_gt"] for r in cat_records if r["iou_with_gt"] is not None]
        mean_iou = float(np.mean(def_ious)) if def_ious else 0.0
        hit_rate = float(np.mean([iou > 0.02 for iou in def_ious])) if def_ious else 0.0

        per_defect = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})
        for r in cat_records:
            d_type = r["defect_type"]
            per_defect[d_type]["total"] += 1
            if r["is_correct"]:
                per_defect[d_type]["correct"] += 1
            per_defect[d_type]["scores"].append(r["anomaly_score"])

        category_summaries[cat] = {
            "total_images": len(cat_records),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "mean_iou": round(mean_iou, 4),
            "defect_hit_rate": round(hit_rate, 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "config": {
                "model_dir": detector.model_dir,
                "use_spatial_prior": detector.use_spatial_prior,
                "prior_mode": detector.prior_mode,
                "score_method": detector.image_score_method,
                "image_threshold": detector.image_threshold,
                "pixel_threshold": detector.pixel_threshold,
                "morphology_kernel": detector.morphology_kernel,
                "min_region_area": detector.min_region_area,
                "border_margin": detector.border_margin
            },
            "per_defect_breakdown": {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": round(v["correct"] / v["total"], 4) if v["total"] > 0 else 0.0,
                    "mean_score": round(float(np.mean(v["scores"])), 4)
                } for k, v in per_defect.items()
            }
        }

        print(f"  [{cat.upper()}] Done: {len(cat_records)} images evaluated. "
              f"Acc: {acc*100:.1f}%, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # Write audit_summary.json
    audit_data = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": "70a22284c491126043f3952a147b06f549c96b5e",
        "total_images_evaluated": len(all_records),
        "categories_evaluated": categories,
        "category_summaries": category_summaries,
        "predictions": all_records
    }
    json_path = OUTPUT_BASE / "audit_summary.json"
    with open(json_path, "w") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\n[OK] audit_summary.json written to {json_path}")

    return audit_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_batch_audit(device)
