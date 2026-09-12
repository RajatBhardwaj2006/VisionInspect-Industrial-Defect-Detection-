import torch
import numpy as np
from pathlib import Path
from src.detection.heatmap import compute_anomaly_map
from src.evaluation.metrics import compute_max_anomaly_score
from src.detection.localization import localize
from tqdm import tqdm
import csv
import json

def pixel_threshold_sweep(model, test_loader, thresholds, device, image_threshold=0.606901):
    """Run inference over the test set for each pixel threshold.

    Returns a list of dicts with metrics per threshold.
    """
    results = []
    # Gather ground‑truth labels and masks for later evaluation
    all_gt_masks = []
    all_is_defective = []
    all_anomaly_maps = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            img, _, _, is_defective, gt_mask = batch
            img = img.to(device)
            recon = model(img)
            anomaly_map = compute_anomaly_map(img, recon).squeeze(0).cpu()  # [H,W]
            all_anomaly_maps.append(anomaly_map)
            all_gt_masks.append(gt_mask.squeeze(0).cpu())
            all_is_defective.append(is_defective.item())
    # Convert list to tensors for vectorised ops if needed
    for thresh in thresholds:
        tp = fp = fn = tn = 0
        for amap, gt_mask, is_def in zip(all_anomaly_maps, all_gt_masks, all_is_defective):
            # Pixel‑level binary prediction
            pred_mask = (amap > thresh).float()
            # Compute TP/FP/FN/TN at pixel level
            tp += int(((pred_mask == 1) & (gt_mask == 1)).sum())
            fp += int(((pred_mask == 1) & (gt_mask == 0)).sum())
            fn += int(((pred_mask == 0) & (gt_mask == 1)).sum())
            tn += int(((pred_mask == 0) & (gt_mask == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        results.append({
            'threshold': thresh,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou
        })
    return results

def save_sweep(results, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'pixel_sweep.csv'
    json_path = out_dir / 'pixel_sweep.json'
    # CSV
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    # JSON
    with json_path.open('w') as f:
        json.dump(results, f, indent=2)
