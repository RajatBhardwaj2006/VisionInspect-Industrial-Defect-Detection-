import os
import sys
import torch
import numpy as np
import csv
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Fix OpenMP duplicate runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.autoencoder import Autoencoder
from src.data.dataset_loader import MVTecTestDataset
from src.detection.heatmap import compute_anomaly_map
from src.detection.localization import localize
from src.evaluation.pixel_sweep import save_sweep

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load baseline model
    model_path = Path("models/bottle/autoencoder.pth")
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        sys.exit(1)
        
    model = Autoencoder()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Baseline model loaded successfully.")
    
    # 2. Load test dataset
    dataset_root = "dataset/mvtec_anomaly_detection"
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category="bottle")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Loaded test dataset with {len(test_dataset)} images.")
    
    # 3. Gather predictions
    image_scores = []
    image_labels = []
    
    # For pixel-level evaluation
    pixel_tp = 0
    pixel_fp = 0
    pixel_fn = 0
    
    all_anomaly_maps = []
    all_gt_masks = []
    all_is_defective = []
    
    # Load config pixel thresh
    from src.utils.config import load_config
    cfg = load_config()
    pixel_threshold = cfg["detection"]["pixel_threshold"]
    image_threshold = cfg["detection"]["image_threshold"]
    
    with torch.no_grad():
        for img, category, defect_type, is_defective, gt_mask in test_loader:
            img = img.to(device)
            recon = model(img)
            amap_tensor = compute_anomaly_map(img, recon).squeeze(0)  # [256, 256]
            amap = amap_tensor.cpu().numpy()
            
            # Image score is the max value of anomaly map
            img_score = float(amap.max())
            image_scores.append(img_score)
            image_labels.append(1 if is_defective.item() else 0)
            
            # Accumulate for pixel sweep
            all_anomaly_maps.append(amap_tensor.cpu())
            all_gt_masks.append(gt_mask.squeeze(0).cpu().numpy())
            all_is_defective.append(is_defective.item())
            
            # Run localize
            regions = localize(amap_tensor, pixel_thresh=pixel_threshold)
            
            # Apply image-level defective classification check with noise filter
            is_def = (img_score > image_threshold) and (len(regions) > 0)
            if is_def:
                top_reg = regions[0]
                if top_reg["total_mass"] < 25.0 and top_reg["score"] < 0.17:
                    is_def = False
                    
            pred_binary = np.zeros((256, 256), dtype=bool)
            if is_def and regions:
                for r in regions:
                    x, y, w, h = r["x"], r["y"], r["width"], r["height"]
                    pred_binary[y:y+h, x:x+w] = True
                    
            gt_binary = (gt_mask.squeeze(0).squeeze(0).numpy() > 0)
            
            if is_defective.item():
                tp = np.logical_and(pred_binary, gt_binary).sum()
                fp = np.logical_and(pred_binary, ~gt_binary).sum()
                fn = np.logical_and(~pred_binary, gt_binary).sum()
                
                pixel_tp += int(tp)
                pixel_fp += int(fp)
                pixel_fn += int(fn)
            else:
                # normal images contribute to FP
                fp = pred_binary.sum()
                pixel_fp += int(fp)
                
    # 4. Compute Image-level AUROC
    image_auroc = roc_auc_score(image_labels, image_scores)
    
    # 5. Compute Pixel-level Metrics
    pixel_precision = pixel_tp / (pixel_tp + pixel_fp) if (pixel_tp + pixel_fp) > 0 else 0.0
    pixel_recall = pixel_tp / (pixel_tp + pixel_fn) if (pixel_tp + pixel_fn) > 0 else 0.0
    pixel_f1 = (2 * pixel_precision * pixel_recall) / (pixel_precision + pixel_recall) if (pixel_precision + pixel_recall) > 0 else 0.0
    pixel_iou = pixel_tp / (pixel_tp + pixel_fp + pixel_fn) if (pixel_tp + pixel_fp + pixel_fn) > 0 else 0.0
    
    print("\n--- Evaluation Metrics ---")
    print(f"Image AUROC:     {image_auroc:.4f}")
    print(f"Pixel Precision: {pixel_precision:.4f}")
    print(f"Pixel Recall:    {pixel_recall:.4f}")
    print(f"Pixel F1-score:  {pixel_f1:.4f}")
    print(f"Pixel IoU:       {pixel_iou:.4f}")
    
    # 6. Save summary report to results/bottle_summary.csv
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "bottle_summary.csv"
    
    with open(summary_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Image AUROC", f"{image_auroc:.6f}"])
        writer.writerow(["Pixel Precision", f"{pixel_precision:.6f}"])
        writer.writerow(["Pixel Recall", f"{pixel_recall:.6f}"])
        writer.writerow(["Pixel F1", f"{pixel_f1:.6f}"])
        writer.writerow(["Pixel IoU", f"{pixel_iou:.6f}"])
    print(f"\nSaved summary to: {summary_path}")
    
    # 7. Run and save pixel sweep
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep_results = []
    for thresh in thresholds:
        tp = fp = fn = tn = 0
        for amap_t, gt_t, is_def_item in zip(all_anomaly_maps, all_gt_masks, all_is_defective):
            regions = localize(amap_t, pixel_thresh=thresh)
            score = float(amap_t.max().item())
            is_d = (score > image_threshold) and (len(regions) > 0)
            if is_d:
                top_reg = regions[0]
                if top_reg["total_mass"] < 25.0 and top_reg["score"] < 0.17:
                    is_d = False
                    
            pred_mask = np.zeros((256, 256), dtype=bool)
            if is_d and regions:
                for r in regions:
                    x, y, w, h = r["x"], r["y"], r["width"], r["height"]
                    pred_mask[y:y+h, x:x+w] = True
            
            gt_binary = (gt_t > 0)
            
            if is_def_item:
                tp += int(np.logical_and(pred_mask, gt_binary).sum())
                fp += int(np.logical_and(pred_mask, ~gt_binary).sum())
                fn += int(np.logical_and(~pred_mask, gt_binary).sum())
                tn += int(np.logical_and(~pred_mask, ~gt_binary).sum())
            else:
                fp += int(pred_mask.sum())
                tn += int((~pred_mask).sum())
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        sweep_results.append({
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
        
    metrics_dir = Path("results/metrics")
    save_sweep(sweep_results, metrics_dir)
    print(f"Saved pixel sweep reports to: {metrics_dir}")

if __name__ == "__main__":
    main()
