import numpy as np
import torch
import cv2
from scipy.ndimage import label
from typing import List, Dict
from src.utils.config import load_config

def postprocess_anomaly_map(anomaly_map: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply lighter Gaussian smoothing to preserve crack edges."""
    amap = anomaly_map.detach().cpu().numpy()
    amap = cv2.GaussianBlur(amap, ksize=(0, 0), sigmaX=sigma)
    return torch.from_numpy(amap).to(anomaly_map.device)

def binary_mask(anomaly_map: torch.Tensor, thresh: float) -> torch.Tensor:
    """Threshold the anomaly map to a binary mask (0/1)."""
    return (anomaly_map > thresh).float()

def morphological_ops(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply mild morphological closing then opening."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy(opened).to(mask.device).float()

def extract_regions(mask: torch.Tensor) -> List[Dict]:
    """Extract connected components and compute region properties."""
    mask_np = mask.detach().cpu().numpy().astype(np.int32)
    labeled, num = label(mask_np)
    regions = []
    for region_idx in range(1, num + 1):
        component = (labeled == region_idx).astype(np.uint8)
        ys, xs = np.where(component)
        if ys.size == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)
        area = int(component.sum())
        centroid = (float(xs.mean()), float(ys.mean()))
        regions.append({
            "x": int(x_min),
            "y": int(y_min),
            "width": width,
            "height": height,
            "area": area,
            "centroid": centroid,
        })
    return regions

def merge_regions(regions: List[Dict], merge_distance: float) -> List[Dict]:
    if not regions or merge_distance <= 0:
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
                
                # Minimum distance between their bounding boxes
                dx = max(0, r1_x1 - r2_x2, r2_x1 - r1_x2)
                dy = max(0, r1_y1 - r2_y2, r2_y1 - r1_y2)
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist <= merge_distance:
                    merge_with = j
                    break
            
            if merge_with != -1:
                r2 = regions[merge_with]
                mx1 = min(r1_x1, r2["x"])
                my1 = min(r1_y1, r2["y"])
                mx2 = max(r1_x2, r2["x"] + r2["width"])
                my2 = max(r1_y2, r2["y"] + r2["height"])
                
                merged_reg = {
                    "x": mx1,
                    "y": my1,
                    "width": mx2 - mx1,
                    "height": my2 - my1,
                    "area": r1["area"] + r2["area"],
                    "centroid": ((r1["centroid"][0] + r2["centroid"][0]) / 2.0, (r1["centroid"][1] + r2["centroid"][1]) / 2.0),
                    "score": (r1.get("score", 0.0) + r2.get("score", 0.0)) / 2.0,
                    "max_val": max(r1.get("max_val", 0.0), r2.get("max_val", 0.0)),
                    "total_mass": r1.get("total_mass", 0.0) + r2.get("total_mass", 0.0)
                }
                regions[i] = merged_reg
                used.add(merge_with)
                merged = True
                break
        
        if merged:
            regions = [regions[idx] for idx in range(len(regions)) if idx not in used]
            
    return regions

def localize(anomaly_map: torch.Tensor, pixel_thresh: float = None) -> List[Dict]:
    """Full localization pipeline prioritizing integrated defect mass (area * score) and region aspect filters."""
    cfg = load_config()
    loc_cfg = cfg["localization"]
    
    if pixel_thresh is None:
        pixel_thresh = cfg["detection"]["pixel_threshold"]
        
    sigma = loc_cfg.get("gaussian_sigma", 1.0)
    kernel_size = loc_cfg.get("morphology_kernel", 3)
    min_area = loc_cfg.get("min_region_area", 30)
    max_area = loc_cfg.get("max_region_area", 3000)
    max_width = loc_cfg.get("max_region_width", 120)
    max_height = loc_cfg.get("max_region_height", 150)
    min_width = loc_cfg.get("min_region_width", 9)
    min_height = loc_cfg.get("min_region_height", 9)
    border_margin = loc_cfg.get("border_margin", 5)
    merge_dist = loc_cfg.get("merge_distance", 10)

    # 1. Gaussian smoothing first to preserve and bridge crack regions
    smooth_map = postprocess_anomaly_map(anomaly_map, sigma=sigma)
    
    # 2. Min-max normalization second to scale the anomaly range
    norm_map = (smooth_map - smooth_map.min()) / (smooth_map.max() - smooth_map.min() + 1e-8)
    
    # 3. Pixel thresholding
    bin_mask = binary_mask(norm_map, pixel_thresh)
    
    # 4. Morphological closing then opening (order swapped to preserve thin cracks)
    clean_mask = morphological_ops(bin_mask, kernel_size=kernel_size)
    
    # 5. Extract connected components
    regions = extract_regions(clean_mask)
    
    # 6. Filter and score regions
    filtered_regions = []
    for reg in regions:
        # Filter border-touching regions
        if (reg['x'] <= border_margin or 
            reg['x'] + reg['width'] >= 256 - border_margin or 
            reg['y'] <= border_margin or 
            reg['y'] + reg['height'] >= 256 - border_margin):
            continue
            
        # Filter regions that are too large, too small, or too narrow (aspect check)
        if reg['area'] > max_area or reg['width'] > max_width or reg['height'] > max_height:
            continue
        if reg['width'] < min_width or reg['height'] < min_height or reg['area'] < min_area:
            continue
            
        # Compute scores on the raw anomaly map
        x, y, w, h = reg["x"], reg["y"], reg["width"], reg["height"]
        region_vals = anomaly_map[y : y + h, x : x + w]
        mean_val = float(region_vals.mean().item())
        max_val = float(region_vals.max().item())
        
        reg["score"] = mean_val
        reg["max_val"] = max_val
        reg["total_mass"] = reg["area"] * mean_val
        
        filtered_regions.append(reg)
        
    # 7. Merge nearby components
    merged_regions = merge_regions(filtered_regions, merge_distance=merge_dist)
    
    # 8. Sort by total mass descending (favors large valid cracks over tiny noise points)
    merged_regions.sort(key=lambda r: r.get("total_mass", 0.0), reverse=True)
    return merged_regions