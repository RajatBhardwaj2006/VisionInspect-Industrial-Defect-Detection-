"""
Phase 3.2 - Localization Pipeline Diagnostic Visualizer
=======================================================
Generates 7-panel diagnostic visualizations for each stage of the pipeline:
  1. Original image
  2. Raw anomaly map (before spatial prior)
  3. Prior-adjusted anomaly map
  4. Gaussian-smoothed map
  5. Binary threshold mask
  6. Morphologically cleaned mask
  7. Final connected components + bounding boxes

Usage:
    python scripts/diagnose_localization_pipeline.py --categories bottle transistor
    python scripts/diagnose_localization_pipeline.py --all

Output: results/phase3/phase3_2_localization/diagnostics/
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import label

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.utils.config import load_config, get_category_config


def get_representative_images(dataset_root: str, category: str) -> List[Tuple[Path, str, bool]]:
    """
    Returns a list of (image_path, label, is_defective) tuples
    covering: normal, obvious defect, small defect (if available).
    """
    ds_root = Path(dataset_root) / category / "test"
    samples = []

    # Always include 2 normal images
    good_dir = ds_root / "good"
    if good_dir.exists():
        normals = sorted(good_dir.glob("*.png"))[:2]
        for p in normals:
            samples.append((p, "normal", False))

    # Include defective images from each defect type (up to 2 per type, max 6 types)
    defect_dirs = sorted([d for d in ds_root.iterdir() if d.is_dir() and d.name != "good"])
    for ddir in defect_dirs[:6]:
        imgs = sorted(ddir.glob("*.png"))
        if imgs:
            samples.append((imgs[0], ddir.name, True))
            if len(imgs) > 1:
                samples.append((imgs[-1], ddir.name + "_alt", True))

    return samples


def run_pipeline_stages(
    img_path: Path,
    model: PatchCoreModelV22,
    use_prior: bool,
    pixel_threshold: float,
    gaussian_sigma: float,
    morph_kernel: int,
    min_area: int,
    border_margin: int,
    device: torch.device
) -> dict:
    """
    Runs image through each pipeline stage and returns all intermediate results.
    """
    original_pil = Image.open(img_path).convert("RGB")
    orig_np = np.array(original_pil)

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    img_tensor = transform(original_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # Stage 1: Raw prediction (without prior)
        raw_score_raw, raw_amap = model._predict_raw(img_tensor)
        raw_amap_np = raw_amap.cpu().numpy()

        # Stage 2: Prior-adjusted map
        if use_prior and model.normal_spatial_prior is not None:
            prior_amap = torch.clamp(raw_amap - model.normal_spatial_prior, min=0.0)
            prior_amap_np = prior_amap.cpu().numpy()
            score = float(prior_amap.max().item())
        else:
            prior_amap_np = raw_amap_np.copy()
            score = raw_score_raw

    # Stage 3: Gaussian smoothing
    if gaussian_sigma > 0:
        smooth_map = cv2.GaussianBlur(prior_amap_np, (0, 0), sigmaX=gaussian_sigma)
    else:
        smooth_map = prior_amap_np.copy()

    # Stage 4: Pixel thresholding
    bin_mask = (smooth_map > pixel_threshold).astype(np.uint8)

    # Stage 5: Morphological cleanup
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Stage 6: Connected components + min area filter
    labeled_arr, num = label(opened)
    clean_mask = np.zeros_like(opened, dtype=bool)
    component_info = []
    for i in range(1, num + 1):
        comp = (labeled_arr == i)
        if comp.sum() >= min_area:
            clean_mask[comp] = True
            ys, xs = np.where(comp)
            component_info.append({
                "area": int(comp.sum()),
                "x_min": int(xs.min()), "x_max": int(xs.max()),
                "y_min": int(ys.min()), "y_max": int(ys.max()),
                "at_border": (xs.min() <= border_margin or xs.max() >= 256 - border_margin or
                               ys.min() <= border_margin or ys.max() >= 256 - border_margin)
            })

    # Stage 7: Bounding boxes (post border-margin filter)
    H, W = clean_mask.shape
    labeled2, num2 = label(clean_mask.astype(np.int32))
    boxes = []
    for idx in range(1, num2 + 1):
        comp = (labeled2 == idx)
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        if (x_min <= border_margin or x_max >= W - border_margin or
                y_min <= border_margin or y_max >= H - border_margin):
            continue
        boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return {
        "original_np": orig_np,
        "raw_amap_np": raw_amap_np,
        "prior_amap_np": prior_amap_np,
        "smooth_map": smooth_map,
        "bin_mask": bin_mask,
        "morph_mask": clean_mask.astype(np.uint8),
        "component_info": component_info,
        "boxes": boxes,
        "score": score,
        "pixel_threshold": pixel_threshold,
    }


def render_diagnostic_figure(
    stages: dict,
    title: str,
    out_path: Path
):
    """
    Renders a 2×4 figure (7 panels) showing all pipeline stages.
    """
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.25)

    orig = stages["original_np"]
    H, W = 256, 256

    def resize_for_display(arr, target=(256, 256)):
        if arr.ndim == 2:
            return cv2.resize(arr.astype(float), (target[1], target[0]), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(arr, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)

    orig_resized = cv2.resize(orig, (W, H))

    # Panel 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_resized)
    ax1.set_title("1. Original", fontsize=10)
    ax1.axis("off")

    # Panel 2: Raw anomaly map
    ax2 = fig.add_subplot(gs[0, 1])
    raw = stages["raw_amap_np"]
    im2 = ax2.imshow(raw, cmap="hot", vmin=0)
    ax2.set_title(f"2. Raw Amap (max={raw.max():.3f})", fontsize=10)
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    ax2.axis("off")

    # Panel 3: Prior-adjusted map
    ax3 = fig.add_subplot(gs[0, 2])
    prior = stages["prior_amap_np"]
    im3 = ax3.imshow(prior, cmap="hot", vmin=0)
    ax3.set_title(f"3. Prior-adj (max={prior.max():.3f})", fontsize=10)
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    ax3.axis("off")

    # Panel 4: Smoothed map
    ax4 = fig.add_subplot(gs[0, 3])
    smooth = stages["smooth_map"]
    pt = stages["pixel_threshold"]
    im4 = ax4.imshow(smooth, cmap="hot", vmin=0)
    ax4.contour(smooth, levels=[pt], colors=['cyan'], linewidths=1)
    ax4.set_title(f"4. Smoothed (thresh={pt:.3f})", fontsize=10)
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    ax4.axis("off")

    # Panel 5: Binary threshold mask
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(stages["bin_mask"], cmap="gray")
    n_px = stages["bin_mask"].sum()
    ax5.set_title(f"5. Binary Mask ({n_px}px above thresh)", fontsize=10)
    ax5.axis("off")

    # Panel 6: Morphological cleaned mask
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(stages["morph_mask"], cmap="gray")
    n_comp = len(stages["component_info"])
    n_border = sum(1 for c in stages["component_info"] if c["at_border"])
    ax6.set_title(f"6. Morph Clean ({n_comp} comp, {n_border} at border)", fontsize=10)
    ax6.axis("off")

    # Panel 7: Final boxes overlaid on original
    ax7 = fig.add_subplot(gs[1, 2])
    overlay = orig_resized.copy()
    for comp in stages["component_info"]:
        if not comp["at_border"]:
            # Green: kept
            cv2.rectangle(overlay, (comp["x_min"], comp["y_min"]),
                          (comp["x_max"], comp["y_max"]), (0, 200, 0), 1)
        else:
            # Red: excluded by border margin
            cv2.rectangle(overlay, (comp["x_min"], comp["y_min"]),
                          (comp["x_max"], comp["y_max"]), (200, 0, 0), 1)
    ax7.imshow(overlay)
    n_boxes = len(stages["boxes"])
    score = stages["score"]
    ax7.set_title(f"7. Final Boxes: {n_boxes} (score={score:.3f})\nGreen=kept, Red=border-excluded", fontsize=9)
    ax7.axis("off")

    # Panel 8: Heatmap overlay
    ax8 = fig.add_subplot(gs[1, 3])
    amap_resized = cv2.resize(smooth, (W, H))
    heatmap = plt.get_cmap('jet')(amap_resized / (amap_resized.max() + 1e-8))[:, :, :3]
    blend = np.clip(0.55 * orig_resized / 255.0 + 0.45 * heatmap, 0, 1)
    ax8.imshow(blend)
    for bx, by, bw, bh in stages["boxes"]:
        rect = plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='yellow', facecolor='none')
        ax8.add_patch(rect)
    ax8.set_title("8. Heatmap + Boxes", fontsize=10)
    ax8.axis("off")

    plt.savefig(out_path, bbox_inches='tight', dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 3.2 Localization Pipeline Diagnostics")
    parser.add_argument("--categories", nargs="+",
                        default=["bottle", "leather", "transistor", "zipper", "screw"])
    parser.add_argument("--dataset-root", type=str, default="dataset/mvtec_anomaly_detection")
    parser.add_argument("--out-dir", type=str,
                        default="results/phase3/phase3_2_localization/diagnostics")
    parser.add_argument("--all", action="store_true", help="Run all 5 categories")
    args = parser.parse_args()

    categories = args.categories if not args.all else \
        ["bottle", "leather", "transistor", "zipper", "screw"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()

    for category in categories:
        cat_cfg = get_category_config(category, cfg)
        model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))

        if not model_dir.exists():
            print(f"[SKIP] Model not found for '{category}' at {model_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"  DIAGNOSING: {category.upper()}")
        print(f"{'='*60}")

        model = PatchCoreModelV22(device=device)
        model.load(str(model_dir))

        use_prior = cat_cfg.get("use_spatial_prior", True)
        pixel_th = cat_cfg.get("pixel_threshold", 1.40)
        gaussian_sigma = cat_cfg.get("gaussian_sigma", 1.20)
        morph_kernel = cat_cfg.get("morphology_kernel", 3)
        min_area = cat_cfg.get("min_region_area", 25)
        border_margin = cat_cfg.get("border_margin", 5)

        # Also load locked thresholds if available (use Phase 3.1 locked values)
        locked_path = Path("results/phase3/final_validation_locked/locked_thresholds.json")
        if locked_path.exists():
            with open(locked_path) as f:
                locked = json.load(f)
            if category in locked:
                pixel_th = locked[category].get("locked_pixel_threshold", pixel_th)
                gaussian_sigma = locked[category].get("locked_gaussian_sigma", gaussian_sigma)
                morph_kernel = locked[category].get("locked_morphology_kernel", morph_kernel)
                min_area = locked[category].get("locked_min_region_area", min_area)
                border_margin = locked[category].get("locked_border_margin", border_margin)

        print(f"  Pixel Threshold: {pixel_th}")
        print(f"  Border Margin:   {border_margin}")
        print(f"  Min Area:        {min_area}")
        print(f"  Morph Kernel:    {morph_kernel}")
        print(f"  Use Prior:       {use_prior}")

        samples = get_representative_images(args.dataset_root, category)
        cat_out = out_dir / category
        cat_out.mkdir(exist_ok=True)

        summary_rows = []

        for img_path, label_str, is_defective in samples:
            try:
                print(f"  Processing: {label_str} / {img_path.name}")
                stages = run_pipeline_stages(
                    img_path=img_path,
                    model=model,
                    use_prior=use_prior,
                    pixel_threshold=pixel_th,
                    gaussian_sigma=gaussian_sigma,
                    morph_kernel=morph_kernel,
                    min_area=min_area,
                    border_margin=border_margin,
                    device=device
                )

                fig_name = f"{category}_{label_str}_{img_path.stem}_diagnostic.png"
                out_path = cat_out / fig_name

                title = (
                    f"Category: {category.upper()} | Sample: {label_str} ({img_path.name}) | "
                    f"Is Defective: {is_defective}\n"
                    f"Phase 3.1 Params: pixel_th={pixel_th:.3f}, sigma={gaussian_sigma}, "
                    f"kernel={morph_kernel}, min_area={min_area}, border_margin={border_margin}"
                )
                render_diagnostic_figure(stages, title, out_path)

                n_border_excluded = sum(1 for c in stages["component_info"] if c["at_border"])
                n_kept = len(stages["boxes"])
                n_total_comp = len(stages["component_info"])
                row = {
                    "category": category,
                    "sample": label_str,
                    "image": img_path.name,
                    "is_defective": is_defective,
                    "score": round(stages["score"], 4),
                    "raw_amap_max": round(float(stages["raw_amap_np"].max()), 4),
                    "prior_amap_max": round(float(stages["prior_amap_np"].max()), 4),
                    "above_threshold_pixels": int(stages["bin_mask"].sum()),
                    "total_components_after_morph": n_total_comp,
                    "border_excluded_components": n_border_excluded,
                    "kept_boxes": n_kept,
                    "would_be_detected": stages["score"] > cat_cfg.get("image_threshold", 1.50)
                }
                summary_rows.append(row)
                print(f"    Score={stages['score']:.4f} | Components={n_total_comp} | "
                      f"Border-excluded={n_border_excluded} | Final boxes={n_kept}")

            except Exception as e:
                print(f"    ERROR for {img_path.name}: {e}")

        # Save summary JSON
        with open(cat_out / "diagnostic_summary.json", "w") as f:
            json.dump(summary_rows, f, indent=4)
        print(f"\n  Diagnostics saved to: {cat_out}")

    print(f"\n{'='*60}")
    print(f"  All diagnostics complete: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
