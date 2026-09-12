import os
import tempfile
from pathlib import Path

import torch
import numpy as np
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.models.patchcore import FeatureExtractor, PatchCoreModel
from src.detection.patchcore_detector import PatchCoreDetector
from src.utils.config import load_config

def test_feature_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor(device=device)
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    patch_feats, (Hp, Wp) = extractor(dummy_input)
    
    assert patch_feats.shape == (2, 1024, 384)
    assert Hp == 32
    assert Wp == 32

def test_patchcore_fit_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreModel(device=device, coreset_sampling_ratio=0.5)
    
    dummy_train = [torch.randn(4, 3, 256, 256) for _ in range(2)]
    model.fit(dummy_train)
    
    assert model.memory_bank is not None
    assert model.memory_bank.shape[1] == 384
    
    dummy_test = torch.randn(1, 3, 256, 256)
    score, amap = model.predict(dummy_test)
    
    assert isinstance(score, float)
    assert amap.shape == (256, 256)
    assert not torch.isnan(amap).any()

def test_patchcore_save_and_load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchCoreModel(device=device, coreset_sampling_ratio=0.5)
    dummy_train = [torch.randn(2, 3, 256, 256)]
    model.fit(dummy_train)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save(tmp_dir)
        assert (Path(tmp_dir) / "memory_bank.pt").exists()
        assert (Path(tmp_dir) / "metadata.json").exists()
        
        loaded_model = PatchCoreModel(device=device)
        loaded_model.load(tmp_dir)
        
        assert loaded_model.memory_bank is not None
        assert loaded_model.memory_bank.shape == model.memory_bank.shape

def test_patchcore_detector_inspect():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path("models/bottle/patchcore")
    if not model_dir.exists():
        pytest.skip("PatchCore trained memory bank not found.")
        
    detector = PatchCoreDetector(category="bottle", model_dir=str(model_dir), device=device)
    test_img = Path("dataset/mvtec_anomaly_detection/bottle/test/good/001.png")
    if not test_img.exists():
        pytest.skip("Test image good/001.png not found.")
        
    result = detector.inspect(str(test_img))
    
    assert "status" in result
    assert "score" in result
    assert "image_threshold" in result
    assert "pixel_threshold" in result
    assert "localized_regions" in result
    assert "saved_path" in result
    assert result["status"] in ["NORMAL", "DEFECTIVE"]
