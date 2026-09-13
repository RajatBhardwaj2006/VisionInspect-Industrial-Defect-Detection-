"""
Phase 3.2 - Spatial Prior Recomputation Script
================================================
Recomputes the p75 robust spatial prior from existing training data.
The memory bank (feature representations) are NOT changed.
Only spatial_prior_p75.pt is added to each model directory.

This script uses ONLY the normal training images — no test data access.

Usage:
    python scripts/recompute_spatial_prior.py --categories transistor zipper screw bottle leather
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.data.dataset_loader import MVTecTrainDataset
from src.utils.config import load_config, get_category_config


def recompute_prior_for_category(
    category: str,
    dataset_root: str,
    device: torch.device
) -> dict:
    """
    Loads existing model, runs all normal training images through it,
    computes both mean (existing) and p75 (new) spatial priors,
    and saves spatial_prior_p75.pt alongside the existing files.
    Does NOT modify memory_bank.pt or metadata.json.
    """
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"\n{'='*60}")
    print(f"  RECOMPUTING SPATIAL PRIOR: {category.upper()}")
    print(f"  Model dir: {model_dir}")
    print(f"{'='*60}")

    use_prior = cat_cfg.get("use_spatial_prior", True)
    if not use_prior:
        print(f"  [SKIP] {category} does not use spatial prior (use_spatial_prior=False)")
        return {"category": category, "status": "skipped_no_prior"}

    # Load model (will load existing memory bank + mean prior)
    model = PatchCoreModelV22(device=device)
    model.load(str(model_dir))

    # Load full normal training dataset
    train_ds = MVTecTrainDataset(dataset_root=dataset_root, category=category)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Normal training images: {len(train_ds)}")

    # Run all normal training images to collect raw anomaly maps
    normal_maps = []
    with torch.no_grad():
        for idx, img_batch in enumerate(train_loader):
            img_batch = img_batch.to(device)
            # Use _predict_raw (no prior subtraction) to get base anomaly maps
            raw_score, raw_amap = model._predict_raw(img_batch)
            normal_maps.append(raw_amap.cpu())
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx+1}/{len(train_ds)} training images...")

    normal_maps_tensor = torch.stack(normal_maps, dim=0)  # [N_train, 256, 256]
    print(f"  Collected {len(normal_maps)} normal anomaly maps, shape: {normal_maps_tensor.shape}")

    # Recompute mean prior (should match existing spatial_prior.pt)
    new_mean_prior = normal_maps_tensor.mean(dim=0)
    print(f"  Mean prior range: [{new_mean_prior.min():.4f}, {new_mean_prior.max():.4f}]")

    # Compare with existing prior
    if model.normal_spatial_prior is not None:
        existing_prior_cpu = model.normal_spatial_prior.cpu()
        diff = (new_mean_prior - existing_prior_cpu).abs().mean().item()
        print(f"  Difference from existing mean prior: {diff:.6f} (should be ~0 if same training data)")

    # Compute p75 prior
    print(f"  Computing p75 prior (this may take a moment)...")
    p75_prior = torch.quantile(normal_maps_tensor, 0.75, dim=0)
    print(f"  P75 prior range: [{p75_prior.min():.4f}, {p75_prior.max():.4f}]")

    # Comparison stats
    mean_val = float(new_mean_prior.mean().item())
    p75_val = float(p75_prior.mean().item())
    suppression_diff = float((p75_prior - new_mean_prior).mean().item())
    print(f"  Mean of mean prior: {mean_val:.4f}")
    print(f"  Mean of p75 prior:  {p75_val:.4f}")
    print(f"  P75 - Mean (avg suppression diff): {suppression_diff:.4f}")

    # Save p75 prior file (does NOT overwrite anything existing)
    p75_path = model_dir / "spatial_prior_p75.pt"
    torch.save(p75_prior, p75_path)
    print(f"  Saved p75 prior to: {p75_path}")

    return {
        "category": category,
        "status": "computed",
        "model_dir": str(model_dir),
        "n_training_images": len(train_ds),
        "mean_prior_avg": round(mean_val, 4),
        "p75_prior_avg": round(p75_val, 4),
        "suppression_diff_mean_to_p75": round(suppression_diff, 4),
        "p75_path": str(p75_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3.2 - Recompute p75 spatial prior")
    parser.add_argument("--categories", nargs="+",
                        default=["bottle", "transistor", "zipper", "screw"])
    parser.add_argument("--dataset-root", type=str, default="dataset/mvtec_anomaly_detection")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    for category in args.categories:
        try:
            result = recompute_prior_for_category(
                category=category,
                dataset_root=args.dataset_root,
                device=device
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR for '{category}': {e}")
            results.append({"category": category, "status": "error", "error": str(e)})

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r.get("status", "unknown")
        if status == "computed":
            print(f"  {r['category']:12s}: P75 prior computed and saved. "
                  f"Suppression diff: {r['suppression_diff_mean_to_p75']:.4f}")
        elif status == "skipped_no_prior":
            print(f"  {r['category']:12s}: Skipped (no spatial prior used)")
        else:
            print(f"  {r['category']:12s}: ERROR — {r.get('error', 'unknown error')}")

    # Save summary
    out_path = Path("results/phase3/phase3_2_localization/calibration/prior_recomputation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
