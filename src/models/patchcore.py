import os
import math
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor(nn.Module):
    """
    Extracts intermediate features from layer2 and layer3 of pretrained ResNet18.
    Concatenates patch features into a uniform spatial resolution (32x32 for 256x256 input).
    """
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        
        # Keep layers up to layer3
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        self.eval()
        self.to(device)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [B, 3, H, W] (e.g. 256x256)
        Output: patch features tensor of shape [B, N_patches, C_total]
                where N_patches = H/8 * W/8 = 32*32 = 1024, C_total = 128 + 256 = 384
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        f_layer2 = self.layer2(x)  # [B, 128, H/8, W/8] e.g. [B, 128, 32, 32]
        f_layer3 = self.layer3(f_layer2)  # [B, 256, H/16, W/16] e.g. [B, 256, 16, 16]
        
        # Resize layer3 features to match layer2 spatial resolution
        target_size = (f_layer2.shape[2], f_layer2.shape[3])
        f_layer3_resized = F.interpolate(f_layer3, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate along channel dimension -> [B, 384, 32, 32]
        f_concat = torch.cat([f_layer2, f_layer3_resized], dim=1)
        
        # Reshape to [B, C, H_p, W_p] -> [B, H_p*W_p, C]
        B, C, Hp, Wp = f_concat.shape
        patch_features = f_concat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        return patch_features, (Hp, Wp)


class PatchCoreModel:
    """
    PatchCore Anomaly Detector model:
    - Feature memory bank extracted from normal training images
    - Coreset reduction for efficient nearest-neighbor search
    - Patch-level and image-level anomaly distance calculation
    """
    def __init__(self, device: Optional[torch.device] = None, coreset_sampling_ratio: float = 0.10):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.feature_extractor = FeatureExtractor(self.device)
        self.memory_bank: Optional[torch.Tensor] = None
        self.patch_grid_shape: Tuple[int, int] = (32, 32)
        
    def fit(self, train_loader: torch.utils.data.DataLoader):
        """
        Extract patch features from all normal training images and build coreset memory bank.
        """
        print("Extracting training patch features for PatchCore memory bank...")
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
                # Flatten batch and spatial patches into single feature list
                feats_flat = patch_feats.reshape(-1, patch_feats.shape[-1])
                all_features.append(feats_flat.cpu())
                
        full_memory_bank = torch.cat(all_features, dim=0)  # [N_patches_total, 384]
        print(f"Full memory bank shape before coreset: {full_memory_bank.shape}")
        
        # Apply coreset sampling
        self.memory_bank = self._coreset_subsampling(full_memory_bank, self.coreset_sampling_ratio)
        self.memory_bank = self.memory_bank.to(self.device)
        print(f"Coreset memory bank constructed with shape: {self.memory_bank.shape}")

    def _coreset_subsampling(self, features: torch.Tensor, ratio: float) -> torch.Tensor:
        """
        Subsamples memory bank using uniform random sampling (or greedy k-center).
        Fast, memory-efficient coreset selection.
        """
        n_samples = max(1, int(len(features) * ratio))
        if n_samples >= len(features):
            return features
            
        # Uniform random sampling for fast coreset construction
        indices = torch.randperm(len(features))[:n_samples]
        return features[indices]

    def predict(self, img_tensor: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Calculates nearest neighbor distances for test image.
        Returns:
            image_score (float): Max anomaly score across patches
            anomaly_map (torch.Tensor): 2D spatial anomaly map of shape [256, 256]
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call fit() or load() before predict().")
            
        img_tensor = img_tensor.to(self.device)
        orig_H, orig_W = img_tensor.shape[2], img_tensor.shape[3]
        
        with torch.no_grad():
            patch_feats, (Hp, Wp) = self.feature_extractor(img_tensor)
            # patch_feats: [1, 1024, 384]
            test_patches = patch_feats.squeeze(0)  # [1024, 384]
            
            # Compute distance matrix: [1024, N_core]
            # Efficient chunked distance computation to save memory
            dists = torch.cdist(test_patches, self.memory_bank, p=2.0)  # [1024, N_core]
            min_dists, _ = torch.min(dists, dim=1)  # [1024]
            
            # Spatial anomaly map at patch grid resolution (Hp, Wp)
            amap_patch = min_dists.reshape(1, 1, Hp, Wp)
            
            # Interpolate to original input resolution (orig_H, orig_W)
            amap_resized = F.interpolate(amap_patch, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            amap_tensor = amap_resized.squeeze(0).squeeze(0)  # [H, W]
            
            # Image score is max value of patch distances
            image_score = float(min_dists.max().item())
            
        return image_score, amap_tensor

    def save(self, dir_path: str):
        """Save memory bank and metadata to directory."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.memory_bank.cpu(), path / "memory_bank.pt")
        meta = {
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "patch_grid_shape": list(self.patch_grid_shape),
            "memory_bank_shape": list(self.memory_bank.shape) if self.memory_bank is not None else None
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
        print(f"PatchCore model saved successfully to: {path}")

    def load(self, dir_path: str):
        """Load memory bank and metadata from directory."""
        path = Path(dir_path)
        bank_path = path / "memory_bank.pt"
        if not bank_path.exists():
            raise FileNotFoundError(f"PatchCore memory bank not found at: {bank_path}")
            
        bank_tensor = torch.load(bank_path, map_location=self.device)
        self.memory_bank = bank_tensor.to(self.device)
        
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.coreset_sampling_ratio = meta.get("coreset_sampling_ratio", self.coreset_sampling_ratio)
                self.patch_grid_shape = tuple(meta.get("patch_grid_shape", (32, 32)))
        print(f"PatchCore model loaded successfully from: {path} (Memory bank size: {self.memory_bank.shape})")
