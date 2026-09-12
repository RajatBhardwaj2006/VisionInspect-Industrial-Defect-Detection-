import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.data.dataset_loader import MVTecTrainDataset
from src.utils.config import load_config, get_category_config

def calibrate_category_normal_split(
    category: str,
    dataset_root: str = "dataset/mvtec_anomaly_detection",
    val_ratio: float = 0.20,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Calibrates image and pixel thresholds using ONLY normal training data.
    ZERO test images and ZERO ground-truth defect masks are accessed.
    """
    category = category.strip().lower()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory for '{category}' not found at: {model_dir}")
        
    print(f"\n========================================================")
    print(f"   CALIBRATING THRESHOLDS: {category.upper()}")
    print(f"   Method: Unsupervised Normal Validation Split ({val_ratio*100:.0f}%)")
    print(f"   Test Set Access: STRICTLY PROHIBITED (Zero Leakage)")
    print(f"========================================================")
    
    # Load model
    model = PatchCoreModelV22(device=device)
    model.load(str(model_dir))
    
    use_prior = cat_cfg.get("use_spatial_prior", True)
    gaussian_sigma = cat_cfg.get("gaussian_sigma", 1.20)
    
    # Load normal training dataset
    full_train_ds = MVTecTrainDataset(dataset_root=dataset_root, category=category)
    total_samples = len(full_train_ds)
    n_val = max(5, int(total_samples * val_ratio))
    
    # Deterministic split: last n_val samples used for calibration
    np.random.seed(42)
    indices = list(range(total_samples))
    # Shuffle deterministically to get an unbiased sample of normal variation
    shuffled_indices = np.random.RandomState(42).permutation(indices)
    val_indices = shuffled_indices[:n_val].tolist()
    
    val_subset = Subset(full_train_ds, val_indices)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    
    print(f"Total Normal Training Images: {total_samples}")
    print(f"Held-out Normal Calibration Images: {len(val_subset)}")
    
    normal_image_scores = []
    normal_pixel_scores = []
    
    with torch.no_grad():
        for img in val_loader:
            img = img.to(device)
            score, amap_tensor = model.predict(img, use_prior=use_prior)
            amap_np = amap_tensor.cpu().numpy()
            
            if gaussian_sigma > 0:
                smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=gaussian_sigma)
            else:
                smooth_map = amap_np
                
            normal_image_scores.append(score)
            # Sample pixels to keep memory bounded
            sampled_pixels = smooth_map.ravel()[::4]
            normal_pixel_scores.extend(sampled_pixels.tolist())
            
    normal_image_scores = np.array(normal_image_scores)
    normal_pixel_scores = np.array(normal_pixel_scores)
    
    # Statistical quantiles
    mean_s = float(np.mean(normal_image_scores))
    std_s = float(np.std(normal_image_scores))
    p95_img = float(np.percentile(normal_image_scores, 95))
    p98_img = float(np.percentile(normal_image_scores, 98))
    p99_img = float(np.percentile(normal_image_scores, 99))
    max_img = float(np.max(normal_image_scores))
    
    # We set image threshold at the 98th percentile of normal variation (+ small margin)
    # to target a ~98% specificity (<=2% false positive rate on normal products).
    calibrated_img_th = round(p98_img * 1.02, 2)
    
    # For pixel threshold, evaluate normal surface pixel intensity quantiles
    p99_px = float(np.percentile(normal_pixel_scores, 99.0))
    p995_px = float(np.percentile(normal_pixel_scores, 99.5))
    p999_px = float(np.percentile(normal_pixel_scores, 99.9))
    
    # We set pixel threshold at the 99.5th percentile of normal surface variations
    calibrated_pixel_th = round(p995_px, 2)
    
    print(f"[Normal Image Scores] Mean={mean_s:.4f}, Std={std_s:.4f}, P95={p95_img:.4f}, P98={p98_img:.4f}, Max={max_img:.4f}")
    print(f"[Normal Pixel Scores] P99={p99_px:.4f}, P99.5={p995_px:.4f}, P99.9={p999_px:.4f}")
    print(f">> LOCKED CALIBRATED PARAMETERS: Image Threshold = {calibrated_img_th}, Pixel Threshold = {calibrated_pixel_th}")
    
    calibration_record = {
        "category": category,
        "calibration_strategy": "unsupervised_normal_split",
        "normal_images_total": total_samples,
        "normal_images_val": n_val,
        "mean_normal_score": round(mean_s, 4),
        "std_normal_score": round(std_s, 4),
        "p95_image_score": round(p95_img, 4),
        "p98_image_score": round(p98_img, 4),
        "p99_image_score": round(p99_img, 4),
        "max_normal_score": round(max_img, 4),
        "p995_pixel_score": round(p995_px, 4),
        "locked_image_threshold": calibrated_img_th,
        "locked_pixel_threshold": calibrated_pixel_th,
        "locked_gaussian_sigma": gaussian_sigma,
        "locked_morphology_kernel": cat_cfg.get("morphology_kernel", 3),
        "locked_min_region_area": cat_cfg.get("min_region_area", 25),
        "locked_border_margin": cat_cfg.get("border_margin", 5),
        "locked_merge_distance": cat_cfg.get("merge_distance", 15.0),
        "use_spatial_prior": use_prior
    }
    
    return calibration_record

def main():
    parser = argparse.ArgumentParser(description="Unsupervised Normal Validation Calibration (No Test Leakage)")
    parser.add_argument("--categories", nargs="+", default=['bottle', 'leather', 'transistor', 'zipper', 'screw'])
    parser.add_argument("--dataset-root", type=str, default="dataset/mvtec_anomaly_detection")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/phase3/final_validation_locked")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_calibrations = {}
    csv_rows = []
    
    for cat in args.categories:
        try:
            record = calibrate_category_normal_split(
                category=cat,
                dataset_root=args.dataset_root,
                val_ratio=args.val_ratio,
                device=device
            )
            all_calibrations[cat] = record
            csv_rows.append(record)
        except Exception as e:
            print(f"Error calibrating category '{cat}': {e}")
            
    # Save locked json
    json_path = out_dir / "locked_thresholds.json"
    with open(json_path, "w") as f:
        json.dump(all_calibrations, f, indent=4)
    print(f"\nLocked parameters saved to: {json_path.resolve()}")
    
    # Save validation CSV
    csv_path = out_dir / "validation_summary.csv"
    if csv_rows:
        headers = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
        print(f"Validation summary CSV saved to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
