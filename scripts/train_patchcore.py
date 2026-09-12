import os
import sys
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.data.dataset_loader import MVTecTrainDataset
from src.utils.config import load_config, get_category_config

def train_category(
    category: str,
    dataset_root: str = "dataset/mvtec_anomaly_detection",
    coreset_ratio: float = 0.10,
    force: bool = False,
    override_spatial_prior: bool = None
):
    category = category.strip().lower()
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"      VISIONINSPECT PATCHCORE v2.3 TRAINING")
    print(f"========================================================")
    print(f"Target Category:       {category}")
    print(f"Compute Device:        {device}")
    print(f"Coreset Sample Ratio:  {coreset_ratio:.2f}")
    
    # Resolve spatial prior setting
    if override_spatial_prior is not None:
        use_spatial_prior = override_spatial_prior
    else:
        use_spatial_prior = cat_cfg.get("use_spatial_prior", True)
    print(f"Use Spatial Prior:     {use_spatial_prior}")
    
    save_dir = Path(f"models/{category}/patchcore_v23")
    
    # Baseline protection for bottle
    if category == "bottle" and save_dir.exists() and not force:
        print(f"\n[PROTECTION] Verified Phase 2.3 baseline for 'bottle' already exists at:")
        print(f"  {save_dir}")
        print(f"Preserving existing verified artifacts (pass --force to override).")
        return save_dir
        
    train_dir = Path(dataset_root) / category / "train" / "good"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
    print(f"\nLoading normal training dataset from: {train_dir}")
    train_dataset = MVTecTrainDataset(dataset_root=dataset_root, category=category)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    print(f"Total Normal Training Images: {len(train_dataset)}")
    
    # Initialize PatchCore V2.2/V2.3 model (ResNet18, layer1+2+3, 64x64 grid)
    model = PatchCoreModelV22(device=device, coreset_sampling_ratio=coreset_ratio)
    
    # Extract features and fit memory bank
    print(f"Extracting multi-scale patch features (layer1+2+3 -> 448 dims)...")
    all_features = []
    with torch.no_grad():
        for batch in train_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device)
            patch_feats, grid_shape = model.feature_extractor(imgs)
            model.patch_grid_shape = grid_shape
            feats_flat = patch_feats.reshape(-1, patch_feats.shape[-1])
            all_features.append(feats_flat.cpu())
            
    full_memory_bank = torch.cat(all_features, dim=0)
    print(f"Full patch pool extracted: {full_memory_bank.shape[0]} patches of dimension {full_memory_bank.shape[1]}")
    
    # Coreset subsampling
    n_samples = max(1, int(len(full_memory_bank) * coreset_ratio))
    indices = torch.randperm(len(full_memory_bank))[:n_samples]
    model.memory_bank = full_memory_bank[indices].to(device)
    print(f"Coreset memory bank constructed with shape: {model.memory_bank.shape}")
    
    # Compute Normal Spatial Background Prior if enabled
    if use_spatial_prior:
        print("Computing Normal Spatial Background Prior across training images...")
        normal_maps = []
        with torch.no_grad():
            for batch in train_loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(device)
                for i in range(imgs.shape[0]):
                    raw_score, amap = model._predict_raw(imgs[i:i+1])
                    normal_maps.append(amap.cpu())
        normal_maps_tensor = torch.stack(normal_maps, dim=0)
        model.normal_spatial_prior = normal_maps_tensor.mean(dim=0).to(device)
        print(f"Spatial background prior computed: range [{model.normal_spatial_prior.min():.4f}, {model.normal_spatial_prior.max():.4f}]")
    else:
        model.normal_spatial_prior = None
        print("Spatial background prior: DISABLED (Texture or non-aligned category)")
        
    # Save artifacts
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.memory_bank.cpu(), save_dir / "memory_bank.pt")
    if model.normal_spatial_prior is not None:
        torch.save(model.normal_spatial_prior.cpu(), save_dir / "spatial_prior.pt")
    elif (save_dir / "spatial_prior.pt").exists():
        # Remove old spatial prior if re-training with prior disabled
        (save_dir / "spatial_prior.pt").unlink()
        
    meta = {
        "coreset_sampling_ratio": coreset_ratio,
        "patch_grid_shape": list(model.patch_grid_shape),
        "memory_bank_shape": list(model.memory_bank.shape),
        "version": "2.3",
        "category": category,
        "use_spatial_prior": use_spatial_prior,
        "n_train_images": len(train_dataset)
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=4)
        
    print(f"\nTraining Successful!")
    print(f"Artifacts saved to: {save_dir.resolve()}")
    print(f"  - memory_bank.pt ({save_dir / 'memory_bank.pt'}): {model.memory_bank.shape}")
    if use_spatial_prior:
        print(f"  - spatial_prior.pt ({save_dir / 'spatial_prior.pt'})")
    print(f"  - metadata.json ({save_dir / 'metadata.json'})")
    print(f"========================================================\n")
    return save_dir

def main():
    parser = argparse.ArgumentParser(description="VisionInspect - Category-Aware PatchCore v2.3 Training")
    parser.add_argument("--category", type=str, required=True, help="MVTec category name (e.g. bottle, leather, transistor, zipper, screw)")
    parser.add_argument("--coreset-ratio", type=float, default=0.10, help="Coreset subsampling ratio (default: 0.10)")
    parser.add_argument("--spatial-prior", action="store_true", default=None, help="Force spatial background prior")
    parser.add_argument("--no-spatial-prior", dest="spatial_prior", action="store_false", help="Disable spatial background prior")
    parser.add_argument("--force", action="store_true", help="Force overwrite even if model already exists")
    
    args = parser.parse_args()
    train_category(
        category=args.category,
        coreset_ratio=args.coreset_ratio,
        force=args.force,
        override_spatial_prior=args.spatial_prior
    )

if __name__ == "__main__":
    main()
