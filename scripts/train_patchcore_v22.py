import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.patchcore_v22 import PatchCoreModelV22
from src.data.dataset_loader import MVTecTrainDataset
from src.utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    category = "bottle"
    dataset_root = "dataset/mvtec_anomaly_detection"
    pc_cfg = cfg.get("patchcore_v22", {})
    coreset_ratio = pc_cfg.get("coreset_sampling_ratio", 0.10)
    save_dir = pc_cfg.get("model_dir", f"models/{category}/patchcore_v22")
    
    print(f"Loading normal training dataset for '{category}'...")
    train_dataset = MVTecTrainDataset(dataset_root=dataset_root, category=category)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    print(f"Total training images: {len(train_dataset)}")
    
    model = PatchCoreModelV22(device=device, coreset_sampling_ratio=coreset_ratio)
    model.fit(train_loader)
    
    model.save(save_dir)
    print(f"PatchCore V2.2 training completed. Artifacts saved to: {save_dir}")

if __name__ == "__main__":
    main()
