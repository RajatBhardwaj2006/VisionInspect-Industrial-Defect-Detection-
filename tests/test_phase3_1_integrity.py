import io
import os
import sys
import json
from pathlib import Path
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.backend import app, SUPPORTED_CATEGORIES
from src.utils.config import load_config, get_category_config

@pytest.fixture
def client():
    return TestClient(app)

def test_api_root_endpoint(client):
    """Test A9: Root GET / endpoint returns status running and docs URL."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["docs"] == "/docs"
    assert "service" in data

def test_custom_image_upload_png(client):
    """Test A1 & A8: Custom uploaded PNG image handled cleanly via multipart/form-data."""
    # Create arbitrary non-dataset image
    img = Image.new("RGB", (100, 100), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "false"},
        files={"file": ("custom_sample.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "bottle"
    assert data["status"] in ["NORMAL", "DEFECTIVE"]
    assert "anomaly_score" in data
    assert "image_threshold" in data
    assert "num_defects" in data
    assert "localized_regions" in data
    assert "inference_time_ms" in data

def test_custom_image_upload_jpeg(client):
    """Test A1: JPEG format support."""
    img = Image.new("RGB", (80, 80), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"category": "leather", "return_visualizations": "false"},
        files={"file": ("custom_leather.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "leather"

def test_custom_image_upload_webp(client):
    """Test A1: WEBP format support."""
    img = Image.new("RGB", (70, 70), color=(50, 150, 250))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"category": "screw", "return_visualizations": "false"},
        files={"file": ("custom_screw.webp", buf, "image/webp")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "screw"

def test_unsupported_file_extension_rejection(client):
    """Test C3: Rejection of unsupported file extension."""
    buf = io.BytesIO(b"dummy text content")
    response = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("report.pdf", buf, "application/pdf")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_empty_file_upload_rejection(client):
    """Test C4: Rejection of empty file upload."""
    buf = io.BytesIO(b"")
    response = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("empty.png", buf, "image/png")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_category_validation_rejection(client):
    """Test C5: Rejection of unconfigured/unsupported category."""
    img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"category": "airplane_parts"},
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 400
    assert "Unsupported category" in response.json()["detail"]

def test_prediction_response_schema_safety(client):
    """Test A8 & C6: No machine secrets, absolute usernames or virtualenv paths leaked."""
    img = Image.new("RGB", (64, 64), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "true"},
        files={"file": ("sample.png", buf, "image/png")}
    )
    assert response.status_code == 200
    raw_text = response.text
    # Ensure local path references and system user directory strings are not leaked
    assert "visioninspect_py312" not in raw_text
    assert "Users\\Rajat" not in raw_text

def test_calibration_does_not_read_test_set():
    """Test B2 & C9: Verify calibration script source code does NOT import or read MVTecTestDataset."""
    calib_script = Path("scripts/calibrate_and_lock_thresholds.py")
    assert calib_script.exists()
    content = calib_script.read_text(encoding="utf-8")
    assert "MVTecTestDataset" not in content
    assert "/test" not in content
    assert "/ground_truth" not in content

def test_locked_thresholds_loaded_in_unbiased_eval():
    """Test B3, B4 & C10: Verify final eval script loads locked parameters without recalibration."""
    eval_script = Path("scripts/run_final_unbiased_eval.py")
    assert eval_script.exists()
    content = eval_script.read_text(encoding="utf-8")
    # Must use locked parameters and must not perform 2D threshold sweep
    assert "locked_params" in content
    assert "threshold_sweep" not in content
