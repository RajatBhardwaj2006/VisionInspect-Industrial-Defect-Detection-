import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from src.data.preprocessing import get_train_transform

class MVTecTrainDataset(Dataset):
    """Dataset for training normal images.

    Loads images from ``<dataset_root>/<category>/train/good/``.
    Returns only the transformed image tensor.
    """
    def __init__(self, dataset_root, category='bottle', transform=None):
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.image_dir = self.dataset_root / category / 'train' / 'good'
        if not self.image_dir.exists():
            raise FileNotFoundError(f'Dataset directory not found: {self.image_dir}')
        self.image_paths = sorted(self.image_dir.glob('*.png'))
        if not self.image_paths:
            raise RuntimeError(f'No PNG images found in: {self.image_dir}')
        self.transform = transform or get_train_transform()
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)

class MVTecTestDataset(Dataset):
    """Dataset for test images (both good and defective).

    Returns (image_tensor, category, defect_type, is_defective, mask_tensor).
    """
    def __init__(self, dataset_root, category='bottle', transform=None, mask_transform=None):
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.transform = transform or get_train_transform()
        self.mask_transform = mask_transform or get_train_transform()
        self.test_dir = self.dataset_root / category / 'test'
        if not self.test_dir.exists():
            raise FileNotFoundError(f'Test directory not found: {self.test_dir}')
        self.samples = []
        for defect_dir in self.test_dir.iterdir():
            if not defect_dir.is_dir():
                continue
            defect_type = defect_dir.name
            for img_path in sorted(defect_dir.glob('*.png')):
                self.samples.append((img_path, defect_type))
        if not self.samples:
            raise RuntimeError(f'No test images found in: {self.test_dir}')
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, defect_type = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        is_defective = defect_type != 'good'
        if is_defective:
            mask_path = (
                self.dataset_root / self.category / 'ground_truth' / defect_type / f"{img_path.stem}_mask.png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(f'Mask not found: {mask_path}')
            mask = Image.open(mask_path).convert('L')
            mask_tensor = self.mask_transform(mask)
        else:
            mask_tensor = torch.zeros_like(img_tensor[:1, :, :])
        return img_tensor, self.category, defect_type, is_defective, mask_tensor