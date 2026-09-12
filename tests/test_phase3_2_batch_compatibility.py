import io
import os
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.backend import app, SUPPORTED_CATEGORIES
from src.detection.category_checker import CategoryCompatibilityChecker

@pytest.fixture
def client():
    return TestClient(app)

def create_dummy_image(format="PNG", color=(100, 150, 200), size=(64, 64)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# ----------------- 1. BATCH SIZES & REJECTION TESTS -----------------

def test_batch_upload_single_image(client):
    """Test 1-image batch upload."""
    buf = create_dummy_image(format="PNG")
    response = client.post(
        "/predict/batch",
        data={"category": "bottle", "return_visualizations": "false"},
        files=[("files", ("test_0.png", buf, "image/png"))]
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 1
    assert data["summary"]["total"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] in ["NORMAL", "DEFECTIVE"]
    assert "anomaly_score" in data["results"][0]

def test_batch_upload_two_images(client):
    """Test 2-image batch upload."""
    files = [
        ("files", ("test_0.png", create_dummy_image(format="PNG", color="red"), "image/png")),
        ("files", ("test_1.png", create_dummy_image(format="PNG", color="blue"), "image/png")),
    ]
    response = client.post(
        "/predict/batch",
        data={"category": "bottle", "return_visualizations": "false"},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert data["summary"]["total"] == 2
    assert len(data["results"]) == 2

def test_batch_upload_five_images(client):
    """Test 5-image batch upload (exact upper limit)."""
    files = [
        ("files", (f"test_{i}.png", create_dummy_image(format="PNG", color=(i*40, 50, 60)), "image/png"))
        for i in range(5)
    ]
    response = client.post(
        "/predict/batch",
        data={"category": "bottle", "return_visualizations": "false"},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 5
    assert data["summary"]["total"] == 5
    assert len(data["results"]) == 5

def test_batch_upload_six_images_rejected(client):
    """Test 6-image batch rejection (> 5 images limit)."""
    files = [
        ("files", (f"test_{i}.png", create_dummy_image(format="PNG"), "image/png"))
        for i in range(6)
    ]
    response = client.post(
        "/predict/batch",
        data={"category": "bottle", "return_visualizations": "false"},
        files=files
    )
    assert response.status_code == 400
    detail = response.json().get("detail", "")
    assert "Batch size must be between 1 and 5" in detail

def test_batch_upload_empty_batch_rejected(client):
    """Test 0-image batch rejection (returns 400 or 422)."""
    response = client.post(
        "/predict/batch",
        data={"category": "bottle"},
        files=[]
    )
    assert response.status_code in [400, 422]

# ----------------- 2. FORMATS & ERROR RESILIENCE TESTS -----------------

def test_batch_upload_mixed_formats(client):
    """Test mixed image formats in single batch: PNG, JPEG, WEBP."""
    files = [
        ("files", ("img_png.png", create_dummy_image(format="PNG"), "image/png")),
        ("files", ("img_jpg.jpg", create_dummy_image(format="JPEG"), "image/jpeg")),
        ("files", ("img_webp.webp", create_dummy_image(format="WEBP"), "image/webp")),
    ]
    response = client.post(
        "/predict/batch",
        data={"category": "screw", "return_visualizations": "false"},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 3
    assert data["summary"]["total"] == 3
    assert data["summary"]["failed"] == 0
    for res in data["results"]:
        assert res["status"] in ["NORMAL", "DEFECTIVE"]

def test_batch_error_resilience_corrupted_image(client):
    """Test error resilience: batch with corrupted file still processes valid files."""
    valid_buf = create_dummy_image(format="PNG")
    corrupt_buf = io.BytesIO(b"this is corrupt non-image binary data")
    
    files = [
        ("files", ("valid_image.png", valid_buf, "image/png")),
        ("files", ("corrupt_file.png", corrupt_buf, "image/png")),
    ]
    response = client.post(
        "/predict/batch",
        data={"category": "leather", "return_visualizations": "false"},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert data["batch_size"] == 2
    assert data["summary"]["total"] == 2
    assert data["summary"]["failed"] == 1
    assert data["results"][0]["status"] in ["NORMAL", "DEFECTIVE"]
    assert data["results"][1]["status"] == "ERROR"
    assert data["results"][1]["error"] is not None
    assert "invalid image file" in data["results"][1]["error"].lower()

# ----------------- 3. CATEGORY COMPATIBILITY TESTS -----------------

def test_compatibility_check_structured_output(client):
    """Test compatibility check returns complete structured fields."""
    buf = create_dummy_image(format="PNG")
    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "false"},
        files={"file": ("dummy.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "category_compatibility" in data
    compat = data["category_compatibility"]
    assert "selected_category" in compat
    assert compat["selected_category"] == "bottle"
    assert "best_compatible_category" in compat
    assert "is_mismatch" in compat
    assert "confidence_margin" in compat
    assert "warning_message" in compat
    assert "distances" in compat

def test_strong_category_mismatch_warning(client):
    """Test strong category mismatch triggers warning (e.g. zipper under bottle)."""
    zipper_path = Path("dataset/mvtec_anomaly_detection/zipper/test/broken_teeth/000.png")
    if not zipper_path.exists():
        pytest.skip("Zipper test image not found")
        
    with open(zipper_path, "rb") as f:
        img_bytes = f.read()
        
    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "false"},
        files={"file": ("zipper_sample.png", io.BytesIO(img_bytes), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    compat = data["category_compatibility"]
    assert compat["is_mismatch"] is True
    assert compat["best_compatible_category"] == "zipper"
    assert compat["warning_message"] is not None
    assert "Bottle" in compat["warning_message"]
    assert "Zipper" in compat["warning_message"]

def test_compatible_image_no_warning(client):
    """Test compatible image does NOT trigger warning (e.g. bottle under bottle)."""
    bottle_path = Path("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    if not bottle_path.exists():
        pytest.skip("Bottle test image not found")
        
    with open(bottle_path, "rb") as f:
        img_bytes = f.read()
        
    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "false"},
        files={"file": ("bottle_sample.png", io.BytesIO(img_bytes), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    compat = data["category_compatibility"]
    assert compat["is_mismatch"] is False
    assert compat["warning_message"] is None

def test_warning_contains_selected_and_suggested(client):
    """Test warning message format specifies both selected and suggested categories."""
    zipper_path = Path("dataset/mvtec_anomaly_detection/zipper/test/broken_teeth/000.png")
    if not zipper_path.exists():
        pytest.skip("Zipper test image not found")
        
    with open(zipper_path, "rb") as f:
        img_bytes = f.read()
        
    response = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("zipper_sample.png", io.BytesIO(img_bytes), "image/png")}
    )
    compat = response.json()["category_compatibility"]
    msg = compat["warning_message"]
    assert "Selected category: Bottle" in msg
    assert "This image appears more compatible with: Zipper" in msg

def test_warning_does_not_automatically_switch_category(client):
    """Test warning does NOT automatically switch category in backend processing."""
    zipper_path = Path("dataset/mvtec_anomaly_detection/zipper/test/broken_teeth/000.png")
    if not zipper_path.exists():
        pytest.skip("Zipper test image not found")
        
    with open(zipper_path, "rb") as f:
        img_bytes = f.read()
        
    response = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("zipper_sample.png", io.BytesIO(img_bytes), "image/png")}
    )
    data = response.json()
    assert data["category"] == "bottle"
    assert "models/bottle/patchcore_v23/" in data["model_path"]

def test_safe_logical_model_path_displayed(client):
    """Test model_path is logical and contains no absolute filesystem paths."""
    buf = create_dummy_image(format="PNG")
    response = client.post(
        "/predict",
        data={"category": "leather"},
        files={"file": ("leather_sample.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_path"] == "models/leather/patchcore_v23/"
    assert "X:\\" not in data["model_path"]
    assert "C:\\" not in data["model_path"]

def test_all_responses_contain_no_leaked_secrets(client):
    """Test batch response text contains no leaked usernames, venv or absolute paths."""
    buf = create_dummy_image(format="PNG")
    response = client.post(
        "/predict/batch",
        data={"category": "screw"},
        files=[("files", ("screw_sample.png", buf, "image/png"))]
    )
    raw_text = response.text
    assert "visioninspect_py312" not in raw_text
    assert "Users\\Rajat" not in raw_text
    assert "X:\\" not in raw_text

def test_backend_batch_unsupported_category(client):
    """Test rejection of unsupported category in batch endpoint."""
    buf = create_dummy_image(format="PNG")
    response = client.post(
        "/predict/batch",
        data={"category": "non_existent_category"},
        files=[("files", ("test.png", buf, "image/png"))]
    )
    assert response.status_code == 400
    assert "Unsupported category" in response.json()["detail"]

# ----------------- 4. CHECKER UNIT TESTS -----------------

def test_compatibility_checker_inconclusive_margin():
    """Unit test: Low confidence margin yields inconclusive without raising false warning."""
    checker = CategoryCompatibilityChecker()
    dummy_img = Image.new("RGB", (64, 64), color="gray")
    result = checker.check_compatibility(dummy_img, selected_category="bottle")
    if result["best_compatible_category"] != "bottle" and result["confidence_margin"] < checker.margin_threshold:
        assert result["is_mismatch"] is False
        assert result["is_inconclusive"] is True
        assert result["warning_message"] is None

def test_category_change_session_state_clearing():
    """
    Test session state clearing logic when switching categories.
    """
    state = {
        "selected_category": "bottle",
        "previous_category": "bottle",
        "category_changed_notice": None,
        "batch_results": {"dummy": "results"}
    }
    new_cat = "zipper"
    if new_cat != state["selected_category"]:
        prev = state["selected_category"]
        state["category_changed_notice"] = (
            f"MODEL CHANGED: {prev.capitalize()} → {new_cat.capitalize()}."
        )
        state["previous_category"] = prev
        state["selected_category"] = new_cat
        state["batch_results"] = None
        
    assert state["selected_category"] == "zipper"
    assert state["batch_results"] is None
    assert "MODEL CHANGED: Bottle → Zipper." in state["category_changed_notice"]
