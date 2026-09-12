import torch
from src.data.dataset_loader import MVTecTrainDataset, MVTecTestDataset

def test_dataset_load():
    """Test train and test dataset loader classes."""
    train_dataset = MVTecTrainDataset(
        dataset_root="dataset/mvtec_anomaly_detection",
        category="bottle"
    )
    assert len(train_dataset) > 0
    img = train_dataset[0]
    assert img.shape == (3, 256, 256)

    test_dataset = MVTecTestDataset(
        dataset_root="dataset/mvtec_anomaly_detection",
        category="bottle"
    )
    assert len(test_dataset) > 0
    img, cat, def_type, is_def, mask = test_dataset[0]
    assert img.shape == (3, 256, 256)
    assert cat == "bottle"

def test_mask_load():
    """Test that ground truth masks are loaded properly with matching dimensions."""
    test_dataset = MVTecTestDataset(
        dataset_root="dataset/mvtec_anomaly_detection",
        category="bottle"
    )
    
    # Find a defective sample to check mask
    def_sample = None
    for img, cat, def_type, is_def, mask in test_dataset:
        if is_def:
            def_sample = (mask, def_type)
            break
            
    assert def_sample is not None
    mask, def_type = def_sample
    assert mask.shape == (1, 256, 256)
    assert mask.sum() > 0  # Mask must have non-zero pixels
