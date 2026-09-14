import os
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import label

from src.models.patchcore_v22 import PatchCoreModelV22
from src.detection.localization import postprocess_anomaly_map
from src.utils.config import load_config, get_category_config

def postprocess_clean_mask(bin_mask: np.ndarray, kernel_size: int = 3, min_area: int = 25) -> np.ndarray:
    """Morphological closing then opening + connected component noise filtering."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    labeled, num = label(opened)
    clean_mask = np.zeros_like(opened, dtype=bool)
    for i in range(1, num + 1):
        comp = (labeled == i)
        if comp.sum() >= min_area:
            clean_mask[comp] = True
    return clean_mask

def extract_tight_regions(clean_mask: np.ndarray, border_margin: int = 5) -> List[Dict]:
    labeled, num = label(clean_mask.astype(np.int32))
    regions = []
    H, W = clean_mask.shape
    for idx in range(1, num + 1):
        comp = (labeled == idx)
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        # Phase 3.2: Strict less-than comparison.
        # border_margin=0 means no exclusion.
        # border_margin=2 means exclude x_min in {0,1}, x_max in {254,255}, etc.
        if (x_min < border_margin or x_max > W - 1 - border_margin or
                y_min < border_margin or y_max > H - 1 - border_margin):
            continue
            
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        area = int(comp.sum())
        
        regions.append({
            "x": x_min,
            "y": y_min,
            "width": w,
            "height": h,
            "area": area,
            "centroid": (float(xs.mean()), float(ys.mean()))
        })
    return regions

def merge_nearby_regions(regions: List[Dict], merge_dist: float = 15.0) -> List[Dict]:
    if not regions or merge_dist <= 0:
        return regions
    merged = True
    while merged:
        merged = False
        used = set()
        for i in range(len(regions)):
            r1 = regions[i]
            r1_x1, r1_y1, r1_x2, r1_y2 = r1["x"], r1["y"], r1["x"] + r1["width"], r1["y"] + r1["height"]
            
            merge_with = -1
            for j in range(i + 1, len(regions)):
                r2 = regions[j]
                r2_x1, r2_y1, r2_x2, r2_y2 = r2["x"], r2["y"], r2["x"] + r2["width"], r2["y"] + r2["height"]
                
                dx = max(0, r1_x1 - r2_x2, r2_x1 - r1_x2)
                dy = max(0, r1_y1 - r2_y2, r2_y1 - r1_y2)
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= merge_dist:
                    merge_with = j
                    break
                    
            if merge_with != -1:
                r2 = regions[merge_with]
                mx1 = min(r1_x1, r2["x"])
                my1 = min(r1_y1, r2["y"])
                mx2 = max(r1_x2, r2["x"] + r2["width"])
                my2 = max(r1_y2, r2["y"] + r2["height"])
                
                regions[i] = {
                    "x": mx1,
                    "y": my1,
                    "width": mx2 - mx1,
                    "height": my2 - my1,
                    "area": r1["area"] + r2["area"],
                    "centroid": ((r1["centroid"][0] + r2["centroid"][0]) / 2.0, (r1["centroid"][1] + r2["centroid"][1]) / 2.0)
                }
                used.add(merge_with)
                merged = True
                break
                
        if merged:
            regions = [regions[idx] for idx in range(len(regions)) if idx not in used]
            
    return regions


class PatchCoreDetectorV23:
    """
    Phase 2.3 / Phase 3.0 Category-Aware PatchCore Detector:
    - Multi-scale 64x64 patch feature map (layer1+2+3)
    - Category-specific normal spatial background prior (optional for textures)
    - Direct absolute distance thresholding & crack-preserving morphology
    - Precise region extraction & tight bounding box localization
    - Dynamic category path resolution & verification
    """
    def __init__(self, category: str = 'bottle', model_dir: Optional[str] = None, device: Optional[torch.device] = None):
        self.category = category.strip().lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = load_config()
        cat_cfg = get_category_config(self.category, self.config)
        
        # Dynamic path resolution: prioritize explicit argument, then category config, then default convention
        default_dir = f"models/{self.category}/patchcore_v23"
        resolved_path = model_dir or cat_cfg.get("model_dir", default_dir)
        dir_path = Path(resolved_path)
        
        if not dir_path.exists():
            raise FileNotFoundError(
                f"PatchCore model directory for category '{self.category}' not found at: {dir_path}. "
                f"Please train it first using: python scripts/train_patchcore.py --category {self.category}"
            )
            
        self.model_dir = str(dir_path)
        self.model = PatchCoreModelV22(device=self.device)
        self.model.load(str(dir_path))
        
        # Metadata inspection for loaded category validation
        meta_path = dir_path / "metadata.json"
        self.loaded_category = self.category
        self.model_version = "2.3"
        self.use_spatial_prior = cat_cfg.get("use_spatial_prior", True)
        
        if meta_path.exists():
            try:
                import json
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.loaded_category = meta.get("category", self.category)
                    self.model_version = meta.get("version", "2.3")
                    # Fallback to metadata default only if not explicitly configured in cat_cfg
                    if "use_spatial_prior" in meta and "use_spatial_prior" not in cat_cfg:
                        self.use_spatial_prior = bool(meta["use_spatial_prior"])
            except Exception:
                pass
                
        self.image_threshold = cat_cfg.get("image_threshold", 1.50)
        self.pixel_threshold = cat_cfg.get("pixel_threshold", 1.40)
        self.gaussian_sigma = cat_cfg.get("gaussian_sigma", 1.20)
        self.morphology_kernel = cat_cfg.get("morphology_kernel", 3)
        self.min_region_area = cat_cfg.get("min_region_area", 25)
        self.border_margin = cat_cfg.get("border_margin", 5)
        self.merge_distance = cat_cfg.get("merge_distance", 15.0)
        self.image_score_method = cat_cfg.get("image_score_method", "max_raw")

        # Phase 3.2: prior_mode — auto-detect p75 if available, else fall back to mean
        self.prior_mode = cat_cfg.get("prior_mode", "mean")
        # Auto-upgrade to p75 if the p75 prior file exists and prior_mode not explicitly set to 'mean'
        p75_path = dir_path / "spatial_prior_p75.pt"
        if p75_path.exists() and self.prior_mode != "mean":
            self.prior_mode = "p75"
        elif p75_path.exists() and cat_cfg.get("prior_mode", "") == "p75":
            self.prior_mode = "p75"
        # Load p75 prior into model if available
        if p75_path.exists() and self.model.normal_spatial_prior_p75 is None:
            import torch as _torch
            self.model.normal_spatial_prior_p75 = _torch.load(
                p75_path, map_location=self.device, weights_only=True
            ).to(self.device)

        print(f"[PatchCoreDetectorV23] Initialized:")
        print(f"  Requested Category:  {self.category}")
        print(f"  Loaded Category:     {self.loaded_category}")
        print(f"  Model Version:       {self.model_version}")
        print(f"  Memory Bank Path:    {self.model_dir}")
        print(f"  Use Spatial Prior:   {self.use_spatial_prior}")
        print(f"  Prior Mode:          {self.prior_mode}")
        print(f"  Score Method:        {self.image_score_method}")
        print(f"  Image Threshold:     {self.image_threshold:.4f}")
        print(f"  Pixel Threshold:     {self.pixel_threshold:.4f}")
        print(f"  Border Margin:       {self.border_margin}")
        print(f"  Min Region Area:     {self.min_region_area}")

    def inspect(self, image_input: Any) -> Dict[str, Any]:
        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"Input image not found at: {image_path}")
            original_pil = Image.open(image_path).convert('RGB')
            defect_type = image_path.parent.name
            stem = image_path.stem
        elif isinstance(image_input, Image.Image):
            original_pil = image_input.convert('RGB')
            defect_type = "custom"
            stem = "input"
        else:
            raise TypeError(f"Expected str, Path, or PIL.Image, got {type(image_input)}")
            
        original_np = np.array(original_pil)
        orig_h, orig_w, _ = original_np.shape

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        
        img_tensor = transform(original_pil).unsqueeze(0).to(self.device)

        # 1. Predict anomaly score & spatial map using category-specific spatial prior setting
        raw_score, amap_tensor = self.model.predict(img_tensor, use_prior=self.use_spatial_prior,
                                                    prior_mode=self.prior_mode)
        amap_np = amap_tensor.detach().cpu().numpy()
        
        # 2. Smooth map
        if self.gaussian_sigma > 0:
            smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=self.gaussian_sigma)
        else:
            smooth_map = amap_np

        # Compute image-level score using configured image_score_method
        if self.image_score_method == "top200":
            score = float(np.sort(smooth_map.flatten())[-200:].mean())
        elif self.image_score_method == "top250":
            score = float(np.sort(smooth_map.flatten())[-250:].mean())
        elif self.image_score_method == "top100":
            score = float(np.sort(smooth_map.flatten())[-100:].mean())
        elif self.image_score_method in ["p99.5", "top_0.5pct"]:
            score = float(np.percentile(smooth_map.flatten(), 99.5))
        elif self.image_score_method in ["p99.9", "top_0.1pct"]:
            score = float(np.percentile(smooth_map.flatten(), 99.9))
        elif self.image_score_method in ["top_1pct", "p99.0"]:
            score = float(np.percentile(smooth_map.flatten(), 99.0))
        else:
            score = raw_score
            
        # 3. Direct absolute distance thresholding
        bin_mask = (smooth_map > self.pixel_threshold)
        
        # 4. Clean mask
        clean_mask = postprocess_clean_mask(bin_mask, kernel_size=self.morphology_kernel, min_area=self.min_region_area)
        
        # 5. Extract & merge regions
        regions = extract_tight_regions(clean_mask, border_margin=self.border_margin)
        merged_regions = merge_nearby_regions(regions, merge_dist=self.merge_distance)
        
        # Score each region
        for reg in merged_regions:
            rx, ry, rw, rh = reg["x"], reg["y"], reg["width"], reg["height"]
            sub_vals = smooth_map[ry : ry + rh, rx : rx + rw]
            reg["score"] = float(sub_vals.mean()) if sub_vals.size > 0 else 0.0
            reg["max_val"] = float(sub_vals.max()) if sub_vals.size > 0 else 0.0
            reg["total_mass"] = float(reg["area"] * reg["score"])
            
        merged_regions.sort(key=lambda r: r.get("total_mass", 0.0), reverse=True)
        
        is_defective = (score > self.image_threshold) and (len(merged_regions) > 0)
        status = "DEFECTIVE" if is_defective else "NORMAL"

        # Scale regions back to original image dimensions
        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0
        
        scaled_regions = []
        for reg in merged_regions:
            scaled_regions.append({
                "x": int(reg["x"] * scale_x),
                "y": int(reg["y"] * scale_y),
                "width": int(reg["width"] * scale_x),
                "height": int(reg["height"] * scale_y),
                "area": int(reg["area"] * scale_x * scale_y),
                "centroid": (float(reg["centroid"][0] * scale_x), float(reg["centroid"][1] * scale_y)),
                "score": reg.get("score", 0.0),
                "max_val": reg.get("max_val", 0.0),
                "total_mass": reg.get("total_mass", 0.0)
            })

        bounding_box = scaled_regions[0] if (is_defective and scaled_regions) else None

        results_dir = Path("results/predictions")
        results_dir.mkdir(parents=True, exist_ok=True)
        saved_path = results_dir / f"patchcore_v23_{self.category}_{defect_type}_{stem}_result.png"

        amap_resized = cv2.resize(smooth_map, (orig_w, orig_h))
        norm_amap = amap_resized / (amap_resized.max() + 1e-8)
        heatmap_rgba = plt.get_cmap('jet')(norm_amap)
        heatmap_rgb = heatmap_rgba[:, :, :3]
        overlay_rgb = (0.5 * original_np / 255.0 + 0.5 * heatmap_rgb)

        # 1. Generate standalone pure heatmap image (in-memory base64)
        buf_hm = io.BytesIO()
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap_rgb)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(buf_hm, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf_hm.seek(0)
        heatmap_base64 = base64.b64encode(buf_hm.read()).decode("utf-8")

        # 2. Generate overlay with bounding boxes
        buf_ov = io.BytesIO()
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(overlay_rgb, 0, 1))
        if is_defective:
            for ridx, r in enumerate(scaled_regions):
                bx, by, bw, bh = r["x"], r["y"], r["width"], r["height"]
                rect = plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(bx, max(0, by - 4), f"R{ridx+1}", color='red', fontsize=10, weight='bold',
                         bbox=dict(boxstyle='square,pad=0.1', facecolor='yellow', alpha=0.7, edgecolor='none'))
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(buf_ov, format='png', bbox_inches='tight', pad_inches=0)
        # Also save to disk for backward compatibility with existing tests
        plt.savefig(saved_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        buf_ov.seek(0)
        overlay_base64 = base64.b64encode(buf_ov.read()).decode("utf-8")

        # Clear 1-2 line plain-language explanations
        decision_margin = round(float(score - self.image_threshold), 4)
        if is_defective:
            explanation = (
                f"An abnormal visual pattern was detected and localized in {len(scaled_regions)} region(s). "
                f"The highlighted areas show where the model found the strongest anomaly response."
            )
            why_explanation = (
                "The model detected visual features that differ significantly from normal patterns. "
                "The highlighted regions indicate the strongest localized anomaly areas."
            )
        elif score > self.image_threshold and len(scaled_regions) == 0:
            explanation = (
                "An elevated anomaly signal was detected, but no localized region satisfied the final defect criteria. "
                "Final inspection status: NORMAL."
            )
            why_explanation = (
                "Although the raw feature score slightly exceeded the image threshold, no connected cluster "
                "passed the minimum defect area and morphology requirements. The part is accepted as NORMAL."
            )
        else:
            explanation = (
                "No significant anomaly region was detected. The image is within the calibrated acceptance criteria."
            )
            why_explanation = (
                "The model found no localized region that satisfied the category's anomaly decision criteria."
            )

        return {
            "category": self.category,
            "loaded_category": self.loaded_category,
            "model_version": self.model_version,
            "model_dir": self.model_dir,
            "use_spatial_prior": self.use_spatial_prior,
            "prior_mode": self.prior_mode,
            "status": status,
            "score": score,
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "decision_margin": decision_margin,
            "bounding_box": bounding_box or "None",
            "localized_regions": scaled_regions if is_defective else [],
            "clean_mask": clean_mask if is_defective else np.zeros((256, 256), dtype=bool),
            "explanation": explanation,
            "why_explanation": why_explanation,
            "heatmap_base64": heatmap_base64,
            "overlay_base64": overlay_base64,
            "visualization_base64": overlay_base64,
            "saved_path": str(saved_path)
        }
