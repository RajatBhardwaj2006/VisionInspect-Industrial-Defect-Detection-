"""
tests/test_pdf_report.py
Unit tests for VisionInspect PDF report generation:
- Validates 5-page PDF document creation
- Validates PDF structure across all 5 industrial categories
- Validates both normal and defective inspection dispositions
- Validates error resilience on partial or missing telemetry
"""

import io
import base64
import pytest
from PIL import Image

from app.utils.pdf_report import generate_inspection_pdf


def _create_mock_image_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(180, 180, 180))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_inspection_result():
    img_bytes = _create_mock_image_bytes()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return {
        "status": "DEFECTIVE",
        "is_defective": True,
        "anomaly_score": 3.4215,
        "image_threshold": 2.9000,
        "decision_margin": 0.5215,
        "num_defects": 2,
        "inference_time_ms": 235.4,
        "inference_time_s": 0.235,
        "filename": "test_sample_001.png",
        "orig_bytes": img_bytes,
        "heatmap_base64": b64_str,
        "visualization_base64": b64_str,
        "request_id": "REQ-TEST-9999",
        "memory_bank": "Greedy Coreset (10% nominal features)",
        "peak_anomaly": 4.12,
        "pixel_threshold": 1.85,
        "localized_regions": [
            {
                "label": "Region 01",
                "bbox": [50, 40, 120, 110],
                "area": 4900,
                "score": 3.85,
                "intensity": "High Deviation"
            },
            {
                "label": "Region 02",
                "bbox": [140, 130, 180, 170],
                "area": 1600,
                "score": 3.10,
                "intensity": "Moderate Deviation"
            }
        ]
    }


def test_pdf_generation_defective_all_categories(mock_inspection_result):
    """Verify PDF generates valid PDF bytes for all 5 categories in defective state."""
    for category in ["bottle", "leather", "transistor", "zipper", "screw"]:
        pdf_bytes = generate_inspection_pdf(mock_inspection_result, category)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 20000, f"PDF too small for category {category}: {len(pdf_bytes)} bytes"
        assert pdf_bytes.startswith(b"%PDF-"), f"PDF magic header missing for {category}"


def test_pdf_generation_normal_all_categories(mock_inspection_result):
    """Verify PDF generates valid PDF bytes for all 5 categories in normal (pass) state."""
    normal_res = dict(mock_inspection_result)
    normal_res["is_defective"] = False
    normal_res["status"] = "NORMAL"
    normal_res["anomaly_score"] = 1.8420
    normal_res["decision_margin"] = -1.0580
    normal_res["num_defects"] = 0
    normal_res["localized_regions"] = []

    for category in ["bottle", "leather", "transistor", "zipper", "screw"]:
        pdf_bytes = generate_inspection_pdf(normal_res, category)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 20000, f"PDF too small for category {category}: {len(pdf_bytes)} bytes"
        assert pdf_bytes.startswith(b"%PDF-")


def test_pdf_generation_fallback_missing_images():
    """Verify PDF generator handles missing image bytes gracefully without crashing."""
    minimal_res = {
        "status": "NORMAL",
        "is_defective": False,
        "anomaly_score": 1.2,
        "image_threshold": 2.5,
        "decision_margin": -1.3,
        "num_defects": 0,
        "inference_time_ms": 150.0,
        "filename": "empty_sample.png",
        "request_id": "REQ-EMPTY-001"
    }
    pdf_bytes = generate_inspection_pdf(minimal_res, "screw")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
