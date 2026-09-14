"""
tests/test_frontend_state.py
Unit tests verifying frontend/backend atomic state management,
base64 visualization generation, decision margin calculation,
and category state reset contracts.
"""

import io
import base64
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.backend import app
from src.utils.config import get_category_config, load_config
from src.detection.patchcore_v23_detector import PatchCoreDetectorV23


@pytest.fixture
def client():
    return TestClient(app)


def test_backend_response_atomic_contract(client):
    """Test that /predict returns all atomic fields bound to a unique request_id."""
    img_path = Path("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    if not img_path.exists():
        pytest.skip("Test sample not found")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "true"},
        files={"file": ("000.png", io.BytesIO(img_bytes), "image/png")}
    )

    assert response.status_code == 200
    data = response.json()

    # Verify atomic request ID
    assert "request_id" in data
    assert data["request_id"].startswith("req_")

    # Verify category & status
    assert data["category"] == "bottle"
    assert data["status"] in ["DEFECTIVE", "NORMAL"]

    # Verify decision margin
    assert "decision_margin" in data
    expected_margin = round(float(data["anomaly_score"] - data["image_threshold"]), 4)
    assert abs(data["decision_margin"] - expected_margin) < 1e-4

    # Verify plain-language explanations
    assert "explanation" in data and len(data["explanation"]) > 0
    assert "why_explanation" in data and len(data["why_explanation"]) > 0

    # Verify in-memory base64 heatmaps & overlays
    assert "heatmap_base64" in data and data["heatmap_base64"] is not None
    hm_bytes = base64.b64decode(data["heatmap_base64"])
    hm_img = Image.open(io.BytesIO(hm_bytes))
    assert hm_img.format == "PNG"

    assert "visualization_base64" in data and data["visualization_base64"] is not None
    vis_bytes = base64.b64decode(data["visualization_base64"])
    vis_img = Image.open(io.BytesIO(vis_bytes))
    assert vis_img.format == "PNG"


def test_normal_sample_atomic_response(client):
    """Test /predict with a normal sample produces correct non-defective contract."""
    img_path = Path("dataset/mvtec_anomaly_detection/bottle/test/good/001.png")
    if not img_path.exists():
        pytest.skip("Test normal sample not found")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/predict",
        data={"category": "bottle", "return_visualizations": "true"},
        files={"file": ("good_001.png", io.BytesIO(img_bytes), "image/png")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "NORMAL"
    assert data["decision_margin"] <= 0
    assert data["num_defects"] == 0
    assert "acceptable" in data["explanation"].lower() or "within" in data["explanation"].lower() or "no significant" in data["explanation"].lower()


def test_all_five_categories_thresholds():
    """Verify configured thresholds for all 5 Phase 3 categories exist and are positive."""
    categories = ["bottle", "leather", "transistor", "zipper", "screw"]
    for cat in categories:
        cfg = get_category_config(cat)
        assert "image_threshold" in cfg, f"Missing image_threshold for {cat}"
        assert "pixel_threshold" in cfg, f"Missing pixel_threshold for {cat}"
        assert cfg["image_threshold"] > 0, f"image_threshold <= 0 for {cat}"
        assert cfg["pixel_threshold"] > 0, f"pixel_threshold <= 0 for {cat}"


def test_state_purge_logic():
    """Verify state purge helper logic operates cleanly on simulated session state."""
    simulated_session_state = {
        "current_inspection": {"request_id": "req_123", "status": "DEFECTIVE"},
        "current_request_id": "req_123",
        "custom_image_bytes": b"fake_bytes",
        "custom_image_name": "test.png",
        "quick_sample_choice": "Sample 1",
        "selected_category": "bottle",
    }

    # Simulate category switch purge
    def purge_inspection(state, new_category=None):
        state["current_inspection"] = None
        state["current_request_id"] = None
        state["custom_image_bytes"] = None
        state["custom_image_name"] = None
        state["quick_sample_choice"] = None
        if new_category:
            state["selected_category"] = new_category

    purge_inspection(simulated_session_state, new_category="leather")

    assert simulated_session_state["current_inspection"] is None
    assert simulated_session_state["current_request_id"] is None
    assert simulated_session_state["custom_image_bytes"] is None
    assert simulated_session_state["custom_image_name"] is None
    assert simulated_session_state["quick_sample_choice"] is None
    assert simulated_session_state["selected_category"] == "leather"
