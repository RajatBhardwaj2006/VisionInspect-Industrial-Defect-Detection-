import os
from pathlib import Path
import torch
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.detection.patchcore_v23_detector import PatchCoreDetectorV23

def test_patchcore_v23_detector_init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path("models/bottle/patchcore_v22")
    if not model_dir.exists():
        pytest.skip("PatchCore model directory not found.")
        
    detector = PatchCoreDetectorV23(category="bottle", model_dir=str(model_dir), device=device)
    assert detector.image_threshold == 1.50
    assert detector.pixel_threshold == 1.40

def test_patchcore_v23_detector_inspect_defective():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path("models/bottle/patchcore_v23")
    if not model_dir.exists():
        pytest.skip("PatchCore model directory not found.")
        
    detector = PatchCoreDetectorV23(category="bottle", model_dir=str(model_dir), device=device)
    test_img = Path("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    if not test_img.exists():
        pytest.skip("Test image broken_large/000.png not found.")
        
    result = detector.inspect(str(test_img))
    assert result["status"] == "DEFECTIVE"
    assert len(result["localized_regions"]) > 0
    assert result["bounding_box"] != "None"
    assert "clean_mask" in result

def test_patchcore_v23_detector_inspect_normal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path("models/bottle/patchcore_v23")
    if not model_dir.exists():
        pytest.skip("PatchCore model directory not found.")
        
    detector = PatchCoreDetectorV23(category="bottle", model_dir=str(model_dir), device=device)
    test_img = Path("dataset/mvtec_anomaly_detection/bottle/test/good/001.png")
    if not test_img.exists():
        pytest.skip("Test image good/001.png not found.")
        
    result = detector.inspect(str(test_img))
    assert result["status"] == "NORMAL"
