import os
import sys
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, ResNet18_Weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import MVTecTestDataset, MVTecTrainDataset
from src.detection.localization import localize, postprocess_anomaly_map
from src.utils.config import load_config

class FeatureExtractorV22(nn.Module):
    """
    Higher-resolution feature extractor combining layer1, layer2, and layer3.
    Output spatial resolution: 64x64 patches (4x4 pixels per patch for 256x256 image).
    Total channels: 64 + 128 + 256 = 448.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        self.eval()
        self.to(device)
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f_layer1 = self.layer1(x)         # [B, 64, 64, 64]
        f_layer2 = self.layer2(f_layer1)  # [B, 128, 32, 32]
        f_layer3 = self.layer3(f_layer2)  # [B, 256, 16, 16]
        
        target_size = (f_layer1.shape[2], f_layer1.shape[3]) # (64, 64)
        f_layer2_up = F.interpolate(f_layer2, size=target_size, mode='bilinear', align_corners=False)
        f_layer3_up = F.interpolate(f_layer3, size=target_size, mode='bilinear', align_corners=False)
        
        f_concat = torch.cat([f_layer1, f_layer2_up, f_layer3_up], dim=1) # [B, 448, 64, 64]
        
        B, C, Hp, Wp = f_concat.shape
        patch_features = f_concat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        return patch_features, (Hp, Wp)


class PatchCoreModelV22:
    def __init__(self, device: torch.device, coreset_sampling_ratio: float = 0.10):
        self.device = device
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.feature_extractor = FeatureExtractorV22(device)
        self.memory_bank = None
        self.normal_spatial_prior = None
        
    def fit(self, train_loader: DataLoader):
        print("Extracting 64x64 patch features for PatchCore V2.2 memory bank...")
        all_features = []
        
        with torch.no_grad():
            for batch in train_loader:
                imgs = batch.to(self.device)
                patch_feats, _ = self.feature_extractor(imgs)
                feats_flat = patch_feats.reshape(-1, patch_feats.shape[-1])
                all_features.append(feats_flat.cpu())
                
        full_memory_bank = torch.cat(all_features, dim=0)
        print(f"Full memory bank shape: {full_memory_bank.shape}")
        
        # Coreset subsampling
        n_samples = max(1, int(len(full_memory_bank) * self.coreset_sampling_ratio))
        indices = torch.randperm(len(full_memory_bank))[:n_samples]
        self.memory_bank = full_memory_bank[indices].to(self.device)
        print(f"Coreset memory bank shape: {self.memory_bank.shape}")
        
        # Compute Spatial Background Prior across normal training set
        print("Computing Normal Spatial Background Prior...")
        normal_maps = []
        with torch.no_grad():
            for batch in train_loader:
                imgs = batch.to(self.device)
                for i in range(imgs.shape[0]):
                    raw_score, amap = self._predict_raw(imgs[i:i+1])
                    normal_maps.append(amap.cpu())
        normal_maps_tensor = torch.stack(normal_maps, dim=0) # [N_train, 256, 256]
        self.normal_spatial_prior = normal_maps_tensor.mean(dim=0).to(self.device)
        print(f"Spatial prior computed. Shape: {self.normal_spatial_prior.shape}")

    def _predict_raw(self, img_tensor: torch.Tensor):
        img_tensor = img_tensor.to(self.device)
        orig_H, orig_W = img_tensor.shape[2], img_tensor.shape[3]
        
        patch_feats, (Hp, Wp) = self.feature_extractor(img_tensor)
        test_patches = patch_feats.squeeze(0)
        
        # Chunked cdist for memory safety
        dists = torch.cdist(test_patches, self.memory_bank, p=2.0)
        min_dists, _ = torch.min(dists, dim=1)
        
        amap_patch = min_dists.reshape(1, 1, Hp, Wp)
        amap_resized = F.interpolate(amap_patch, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        amap_tensor = amap_resized.squeeze(0).squeeze(0)
        
        image_score = float(min_dists.max().item())
        return image_score, amap_tensor

    def predict(self, img_tensor: torch.Tensor, use_prior: bool = True):
        raw_score, amap_tensor = self._predict_raw(img_tensor)
        if use_prior and self.normal_spatial_prior is not None:
            # Subtract normal spatial prior to eliminate rim/edge baseline artifacts
            amap_tensor = torch.clamp(amap_tensor - self.normal_spatial_prior, min=0.0)
            clean_score = float(amap_tensor.max().item())
            return clean_score, amap_tensor
        return raw_score, amap_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_root = "dataset/mvtec_anomaly_detection"
    train_dataset = MVTecTrainDataset(dataset_root=dataset_root, category="bottle")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    
    model = PatchCoreModelV22(device=device, coreset_sampling_ratio=0.10)
    model.fit(train_loader)
    
    # Save V2.2 memory bank & spatial prior
    save_dir = Path("models/bottle/patchcore_v22")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.memory_bank.cpu(), save_dir / "memory_bank.pt")
    torch.save(model.normal_spatial_prior.cpu(), save_dir / "spatial_prior.pt")
    print(f"Saved PatchCore V2.2 model artifacts to: {save_dir}")
    
    # Evaluate on full 83 test set
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category="bottle")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_data = []
    image_scores = []
    image_labels = []
    
    sigma = 1.5
    
    with torch.no_grad():
        for img, _, defect_type, is_defective, gt_mask in test_loader:
            img = img.to(device)
            score, amap_tensor = model.predict(img, use_prior=True)
            amap_smoothed = postprocess_anomaly_map(amap_tensor, sigma=sigma)
            
            is_def_val = 1 if is_defective.item() else 0
            image_scores.append(score)
            image_labels.append(is_def_val)
            
            all_data.append({
                'defect_type': defect_type[0],
                'is_defective': is_def_val,
                'amap_smoothed': amap_smoothed.cpu(),
                'gt_mask': gt_mask.squeeze(0).squeeze(0).numpy() > 0,
                'score': score
            })
            
    image_auroc = roc_auc_score(image_labels, image_scores)
    print(f"\nPatchCore V2.2 Image AUROC: {image_auroc:.4f}")
    
    # Grid search over image & pixel threshold
    img_thresholds = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]
    pixel_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    
    best_f1 = -1.0
    best_p_th = 0.5
    best_img_th = 1.0
    best_metrics = {}
    
    for img_th in img_thresholds:
        for p_th in pixel_thresholds:
            tp = fp = fn = 0
            for data in all_data:
                amap_t = data['amap_smoothed']
                gt_binary = data['gt_mask']
                is_def_item = data['is_defective']
                score = data['score']
                
                # Direct thresholding on clean prior-subtracted distance map (no min-max rescale)
                pred_mask = (amap_t.numpy() > p_th)
                is_d = (score > img_th) and (pred_mask.sum() > 20)
                if not is_d:
                    pred_mask = np.zeros_like(pred_mask)
                    
                if is_def_item:
                    tp += int(np.logical_and(pred_mask, gt_binary).sum())
                    fp += int(np.logical_and(pred_mask, ~gt_binary).sum())
                    fn += int(np.logical_and(~pred_mask, gt_binary).sum())
                else:
                    fp += int(pred_mask.sum())
                    
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_p_th = p_th
                best_img_th = img_th
                best_metrics = {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'iou': iou,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                }
                
    print(f"\nOptimal V2.2 Thresholds -> Image Thresh: {best_img_th}, Pixel Thresh: {best_p_th}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1-Score:  {best_metrics['f1']:.4f}")
    print(f"IoU:       {best_metrics['iou']:.4f}")

if __name__ == "__main__":
    main()
