import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractorV22(nn.Module):
    """
    Higher-resolution feature extractor combining ResNet18 layer1, layer2, and layer3.
    Output spatial resolution: 64x64 patch grid (4x4 pixel resolution per patch for 256x256 image).
    Total channel dimension: 64 + 128 + 256 = 448.
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
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f_layer1 = self.layer1(x)         # [B, 64, 64, 64]
        f_layer2 = self.layer2(f_layer1)  # [B, 128, 32, 32]
        f_layer3 = self.layer3(f_layer2)  # [B, 256, 16, 16]
        
        target_size = (f_layer1.shape[2], f_layer1.shape[3])  # (64, 64)
        f_layer2_up = F.interpolate(f_layer2, size=target_size, mode='bilinear', align_corners=False)
        f_layer3_up = F.interpolate(f_layer3, size=target_size, mode='bilinear', align_corners=False)
        
        f_concat = torch.cat([f_layer1, f_layer2_up, f_layer3_up], dim=1)  # [B, 448, 64, 64]
        
        B, C, Hp, Wp = f_concat.shape
        patch_features = f_concat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        return patch_features, (Hp, Wp)


class PatchCoreModelV22:
    """
    Crack-sensitive PatchCore V2.2 Model:
    - 64x64 multi-scale patch feature memory bank (448 dimensions)
    - Normal Spatial Background Prior subtraction to eliminate bottle rim/edge artifacts
    - Coreset sampling & fast GPU Euclidean distance calculation
    - Phase 3.2: Optional p75 robust prior mode for softer structural suppression
    """
    def __init__(self, device: Optional[torch.device] = None, coreset_sampling_ratio: float = 0.10):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.feature_extractor = FeatureExtractorV22(self.device)
        self.memory_bank: Optional[torch.Tensor] = None
        self.normal_spatial_prior: Optional[torch.Tensor] = None
        self.normal_spatial_prior_p75: Optional[torch.Tensor] = None  # Phase 3.2: p75 robust prior
        self.patch_grid_shape: Tuple[int, int] = (64, 64)

    def fit(self, train_loader: torch.utils.data.DataLoader):
        """Extract features from all normal training images and build memory bank + spatial prior."""
        print("Extracting 64x64 patch features for PatchCore V2.2 memory bank...")
        all_features = []
        
        with torch.no_grad():
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch
                imgs = imgs.to(self.device)
                patch_feats, grid_shape = self.feature_extractor(imgs)
                self.patch_grid_shape = grid_shape
                feats_flat = patch_feats.reshape(-1, patch_feats.shape[-1])
                all_features.append(feats_flat.cpu())
                
        full_memory_bank = torch.cat(all_features, dim=0)  # [N_patches, 448]
        print(f"Full memory bank shape: {full_memory_bank.shape}")
        
        # Coreset subsampling
        n_samples = max(1, int(len(full_memory_bank) * self.coreset_sampling_ratio))
        indices = torch.randperm(len(full_memory_bank))[:n_samples]
        self.memory_bank = full_memory_bank[indices].to(self.device)
        print(f"Coreset memory bank constructed with shape: {self.memory_bank.shape}")
        
        # Compute Normal Spatial Background Prior
        print("Computing Normal Spatial Background Prior...")
        normal_maps = []
        with torch.no_grad():
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch
                imgs = imgs.to(self.device)
                for i in range(imgs.shape[0]):
                    raw_score, amap = self._predict_raw(imgs[i:i+1])
                    normal_maps.append(amap.cpu())
        normal_maps_tensor = torch.stack(normal_maps, dim=0)  # [N_train, 256, 256]
        self.normal_spatial_prior = normal_maps_tensor.mean(dim=0).to(self.device)
        print(f"Spatial background prior computed with mean range: [{self.normal_spatial_prior.min():.4f}, {self.normal_spatial_prior.max():.4f}]")

        # Phase 3.2: Compute p75 robust prior (softer structural suppression)
        # Quantile computation on CPU to avoid memory issues
        normal_maps_cpu = normal_maps_tensor.cpu()
        self.normal_spatial_prior_p75 = torch.quantile(normal_maps_cpu, 0.75, dim=0).to(self.device)
        print(f"P75 robust prior range: [{self.normal_spatial_prior_p75.min():.4f}, {self.normal_spatial_prior_p75.max():.4f}]")

    def _predict_raw(self, img_tensor: torch.Tensor) -> Tuple[float, torch.Tensor]:
        img_tensor = img_tensor.to(self.device)
        orig_H, orig_W = img_tensor.shape[2], img_tensor.shape[3]
        
        patch_feats, (Hp, Wp) = self.feature_extractor(img_tensor)
        test_patches = patch_feats.squeeze(0)
        
        # Chunked cdist for memory efficiency
        dists = torch.cdist(test_patches, self.memory_bank, p=2.0)
        min_dists, _ = torch.min(dists, dim=1)
        
        amap_patch = min_dists.reshape(1, 1, Hp, Wp)
        amap_resized = F.interpolate(amap_patch, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        amap_tensor = amap_resized.squeeze(0).squeeze(0)
        
        image_score = float(min_dists.max().item())
        return image_score, amap_tensor

    def predict(self, img_tensor: torch.Tensor, use_prior: bool = True,
                prior_mode: str = 'mean') -> Tuple[float, torch.Tensor]:
        """
        Predicts anomaly score and spatial map for test image.

        Args:
            img_tensor: Input image tensor [1, 3, H, W]
            use_prior: If True, subtracts normal spatial prior to suppress structural artifacts.
            prior_mode: 'mean' (Phase 3.1 default) or 'p75' (Phase 3.2 softer prior).
                - 'mean': Subtracts per-pixel mean of normal training maps.
                - 'p75': Subtracts per-pixel 75th percentile of normal training maps.
                          Less aggressive suppression; preserves more defect signal at structural borders.
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call fit() or load() before predict().")

        raw_score, amap_tensor = self._predict_raw(img_tensor)
        if use_prior:
            if prior_mode == 'p75' and self.normal_spatial_prior_p75 is not None:
                amap_subbed = torch.clamp(amap_tensor - self.normal_spatial_prior_p75, min=0.0)
                score = float(amap_subbed.max().item())
                return score, amap_subbed
            elif self.normal_spatial_prior is not None:
                amap_subbed = torch.clamp(amap_tensor - self.normal_spatial_prior, min=0.0)
                score = float(amap_subbed.max().item())
                return score, amap_subbed
        return raw_score, amap_tensor

    def save(self, dir_path: str):
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.memory_bank.cpu(), path / "memory_bank.pt")
        if self.normal_spatial_prior is not None:
            torch.save(self.normal_spatial_prior.cpu(), path / "spatial_prior.pt")
        # Phase 3.2: Save p75 prior if computed
        if self.normal_spatial_prior_p75 is not None:
            torch.save(self.normal_spatial_prior_p75.cpu(), path / "spatial_prior_p75.pt")

        meta = {
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "patch_grid_shape": list(self.patch_grid_shape),
            "memory_bank_shape": list(self.memory_bank.shape) if self.memory_bank is not None else None,
            "version": "2.2",
            "has_p75_prior": self.normal_spatial_prior_p75 is not None
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
        print(f"PatchCore V2.2 model saved successfully to: {path}")

    def load(self, dir_path: str):
        path = Path(dir_path)
        bank_path = path / "memory_bank.pt"
        if not bank_path.exists():
            raise FileNotFoundError(f"PatchCore memory bank not found at: {bank_path}")

        self.memory_bank = torch.load(bank_path, map_location=self.device, weights_only=True).to(self.device)

        prior_path = path / "spatial_prior.pt"
        if prior_path.exists():
            self.normal_spatial_prior = torch.load(prior_path, map_location=self.device, weights_only=True).to(self.device)

        # Phase 3.2: Load p75 prior if available (backward compatible)
        prior_p75_path = path / "spatial_prior_p75.pt"
        if prior_p75_path.exists():
            self.normal_spatial_prior_p75 = torch.load(prior_p75_path, map_location=self.device, weights_only=True).to(self.device)

        meta_path = path / "metadata.json"
        version_str = "2.2"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.coreset_sampling_ratio = meta.get("coreset_sampling_ratio", self.coreset_sampling_ratio)
                self.patch_grid_shape = tuple(meta.get("patch_grid_shape", (64, 64)))
                version_str = meta.get("version", "2.2")
        has_p75 = self.normal_spatial_prior_p75 is not None
        print(f"PatchCore V{version_str} model loaded from: {path} (Memory bank: {self.memory_bank.shape}, p75_prior: {has_p75})")
