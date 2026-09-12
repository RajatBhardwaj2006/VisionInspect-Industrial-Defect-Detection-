import os
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.patchcore import PatchCoreModel, FeatureExtractor
from src.data.dataset_loader import MVTecTestDataset, MVTecTrainDataset
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load existing PatchCore model
    model_dir = "models/bottle/patchcore"
    model = PatchCoreModel(device=device)
    model.load(model_dir)
    
    # 2. Compute Normal Background Distance Prior across normal training images
    dataset_root = "dataset/mvtec_anomaly_detection"
    train_dataset = MVTecTrainDataset(dataset_root=dataset_root, category="bottle")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    
    print("Computing normal background distance prior...")
    normal_maps = []
    with torch.no_grad():
        for batch in train_loader:
            imgs = batch.to(device)
            for i in range(imgs.shape[0]):
                img_single = imgs[i:i+1]
                _, amap = model.predict(img_single)
                normal_maps.append(amap.cpu().numpy())
                
    normal_maps_np = np.stack(normal_maps, axis=0)  # [N_train, 256, 256]
    mean_normal_map = normal_maps_np.mean(axis=0)    # [256, 256]
    std_normal_map = normal_maps_np.std(axis=0)      # [256, 256]
    
    print(f"Normal map stats -> Min: {mean_normal_map.min():.4f}, Max: {mean_normal_map.max():.4f}, Mean: {mean_normal_map.mean():.4f}")
    
    # 3. Test on diagnostic subset
    test_dataset = MVTecTestDataset(dataset_root=dataset_root, category="bottle")
    
    target_samples = [
        ("good", "001.png"),
        ("good", "002.png"),
        ("broken_large", "000.png"),
        ("broken_small", "000.png"),
        ("contamination", "000.png"),
        ("contamination", "002.png")
    ]
    
    out_dir = Path("results/phase2_2_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for defect_type, img_name in target_samples:
        img_path = Path(dataset_root) / "bottle" / "test" / defect_type / img_name
        gt_path = Path(dataset_root) / "bottle" / "ground_truth" / defect_type / img_name.replace(".png", "_mask.png")
        has_gt = gt_path.exists()
        
        img_pil = Image.open(img_path).convert("RGB")
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score, amap = model.predict(img_tensor)
            amap_np = amap.cpu().numpy()
            
        # Subtract background prior
        amap_subbed = np.maximum(0, amap_np - mean_normal_map)
        
        # Z-score normalization using normal prior std
        amap_zscore = (amap_np - mean_normal_map) / (std_normal_map + 1e-4)
        amap_zscore = np.maximum(0, amap_zscore)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Original Image
        axes[0, 0].imshow(img_pil)
        axes[0, 0].set_title(f"Original ({defect_type}/{img_name})")
        axes[0, 0].axis("off")
        
        # 2. Raw Anomaly Map
        im1 = axes[0, 1].imshow(amap_np, cmap="jet")
        axes[0, 1].set_title(f"Raw Map (Max: {amap_np.max():.2f})")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. Subtracted Prior Map
        im2 = axes[0, 2].imshow(amap_subbed, cmap="jet")
        axes[0, 2].set_title(f"Prior-Subtracted (Max: {amap_subbed.max():.2f})")
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2])
        
        # 4. Z-Score Anomaly Map
        im3 = axes[1, 0].imshow(amap_zscore, cmap="jet")
        axes[1, 0].set_title(f"Z-Score Map (Max: {amap_zscore.max():.2f})")
        axes[1, 0].axis("off")
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 5. Thresholded Binary Mask (Prior-subtracted at thresh 1.0)
        bin_mask = (amap_subbed > 0.8).astype(np.uint8)
        axes[1, 1].imshow(bin_mask, cmap="gray")
        axes[1, 1].set_title("Subtracted Mask (>0.8)")
        axes[1, 1].axis("off")
        
        # 6. Ground Truth Mask
        if has_gt:
            gt_pil = Image.open(gt_path).convert("L")
            gt_np = cv2.resize(np.array(gt_pil), (256, 256))
            axes[1, 2].imshow(gt_np, cmap="gray")
            axes[1, 2].set_title("Ground Truth Mask")
        else:
            axes[1, 2].imshow(np.zeros((256, 256)), cmap="gray")
            axes[1, 2].set_title("No GT (Normal)")
        axes[1, 2].axis("off")
        
        plt.tight_layout()
        save_path = out_dir / f"diag_{defect_type}_{img_name}"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved diagnostic plot to: {save_path}")

if __name__ == "__main__":
    main()
