import os
import tempfile
from pathlib import Path

import torch
import numpy as np
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models.patchcore_v22 import FeatureExtractorV22, PatchCoreModelV22
from src.detection.patchcore_v22_detector import PatchCoreDetectorV22

def test_feature_extractor_v22():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractorV22(device=device)
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    patch_feats, (Hp, Wp) = extractor(dummy_input)
    
    assert patch_feats.shape == (2, 4096, 448)
    assert Hp == 64
    assert Wp == 64

def test_patchcore_v22_fit_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreModelV22(device=device, coreset_sampling_ratio=0.5)
    
    dummy_train = [torch.randn(4, 3, 256, 256) for _ in range(2)]
    model.fit(dummy_train)
    
    assert model.memory_bank is not None
    assert model.memory_bank.shape[1] == 448
    assert model.normal_spatial_prior is not None
    assert model.normal_spatial_prior.shape == (256, 256)
    
    dummy_test = torch.randn(1, 3, 256, 256)
    score, amap = model.predict(dummy_test, use_prior=True)
    
    assert isinstance(score, float)
    assert amap.shape == (256, 256)
    assert not torch.isnan(amap).any()

def test_patchcore_v22_save_and_load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreModelV22(device=device, coreset_sampling_ratio=0.5)
    dummy_train = [torch.randn(2, 3, 256, 256)]
    model.fit(dummy_train)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        assert (Path(tmp_dir) / "memory_bank.pt").exists()
        assert (Path(tmp_dir) / "spatial_prior.pt").exists()
        assert (Path(tmp_dir) / "metadata.json").exists()
        
        loaded_model = PatchCoreModelV22(device=device)
        loaded_model.load(tmp_dir)
        
        assert loaded_model.memory_bank is not None
        assert loaded_model.normal_spatial_prior is not None
        assert loaded_model.memory_bank.shape == model.memory_bank.shape

def test_patchcore_v22_detector_inspect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path("models/bottle/patchcore_v22")
    if not model_dir.exists():
        pytest.skip("PatchCore V2.2 trained model directory not found.")
        
    detector = PatchCoreDetectorV22(category="bottle", model_dir=str(model_dir), device=device)
    test_img = Path("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    if not test_img.exists():
        pytest.skip("Test image broken_large/000.png not found.")
        
    result = detector.inspect(str(test_img))
    
    assert "status" in result
    assert "score" in result
    assert "image_threshold" in result
    assert "pixel_threshold" in result
    assert "localized_regions" in result
    assert "saved_path" in result
    assert result["status"] in ["NORMAL", "DEFECTIVE"]
