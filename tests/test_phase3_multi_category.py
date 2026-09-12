import io
import os
import sys
from pathlib import Path
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_category_config
from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
from app.backend import app

def test_category_config_loader():
    cfg = load_config()
    assert "categories" in cfg
    
    # Bottle
    bottle_cfg = get_category_config("bottle", cfg)
    assert bottle_cfg["use_spatial_prior"] is True
    assert bottle_cfg["image_threshold"] == 1.50
    assert bottle_cfg["pixel_threshold"] == 1.40
    
    # Leather (texture)
    leather_cfg = get_category_config("leather", cfg)
    assert leather_cfg["use_spatial_prior"] is False
    assert leather_cfg["image_threshold"] == 2.20
    assert leather_cfg["pixel_threshold"] == 2.71
    
    # Screw
    screw_cfg = get_category_config("screw", cfg)
    assert screw_cfg["use_spatial_prior"] is True
    assert screw_cfg["image_threshold"] == 2.28
    assert screw_cfg["pixel_threshold"] == 2.00
    
    # Unconfigured fallback
    unknown_cfg = get_category_config("unknown_item", cfg)
    assert unknown_cfg["use_spatial_prior"] is True  # default for non-texture

def test_category_detector_initialization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Leather detector loads with use_spatial_prior=False
    leather_model = Path("models/leather/patchcore_v23")
    if leather_model.exists():
        det_leather = PatchCoreDetectorV23(category="leather", device=device)
        assert det_leather.category == "leather"
        assert det_leather.use_spatial_prior is False
        assert det_leather.image_threshold == 2.20
        assert det_leather.pixel_threshold == 2.71
        
    # Screw detector loads with use_spatial_prior=True
    screw_model = Path("models/screw/patchcore_v23")
    if screw_model.exists():
        det_screw = PatchCoreDetectorV23(category="screw", device=device)
        assert det_screw.category == "screw"
        assert det_screw.use_spatial_prior is True
        assert det_screw.image_threshold == 2.28
        assert det_screw.pixel_threshold == 2.00

def test_untrained_category_raises():
    with pytest.raises(FileNotFoundError):
        PatchCoreDetectorV23(category="nonexistent_product_xyz")

def test_fastapi_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "supported_categories" in data
    assert "bottle" in data["supported_categories"]
    assert "leather" in data["supported_categories"]
    assert "screw" in data["supported_categories"]
    assert "device" in data

def test_fastapi_categories():
    client = TestClient(app)
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    cats = {c["category"]: c for c in data["categories"]}
    assert "bottle" in cats
    assert "leather" in cats
    assert cats["leather"]["use_spatial_prior"] is False
    assert cats["bottle"]["use_spatial_prior"] is True

def test_fastapi_predict_invalid_category():
    client = TestClient(app)
    # Create tiny dummy image
    img = Image.new("RGB", (64, 64), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"category": "invalid_category"},
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 400
    assert "Unsupported category" in response.json()["detail"]

def test_fastapi_predict_invalid_file():
    client = TestClient(app)
    # Test 1: Non-image extension
    buf = io.BytesIO(b"this is not an image file")
    response = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("test.txt", buf, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"] or "Invalid image" in response.json()["detail"]
    
    # Test 2: Invalid/corrupted image bytes with .png extension
    buf2 = io.BytesIO(b"corrupted binary data")
    response2 = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("test.png", buf2, "image/png")}
    )
    assert response2.status_code == 400
    assert "Invalid image" in response2.json()["detail"]

def test_fastapi_predict_end_to_end():
    client = TestClient(app)
    sample_img_path = Path("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    if not sample_img_path.exists():
        pytest.skip("Test sample image not found.")
        
    with open(sample_img_path, "rb") as f:
        response = client.post(
            "/predict",
            data={"category": "bottle", "return_visualizations": "true"},
            files={"file": ("000.png", f, "image/png")}
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "bottle"
    assert data["status"] == "DEFECTIVE"
    assert data["is_defective"] is True
    assert data["anomaly_score"] > data["image_threshold"]
    assert data["num_defects"] > 0
    assert len(data["defect_regions"]) > 0
    assert data["visualization_base64"] is not None
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] > 0
