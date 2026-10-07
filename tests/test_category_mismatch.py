"""
tests/test_category_mismatch.py
===============================
Tests verifying:
1. Category mismatch detection between selected model and uploaded image
2. Dialog helper contracts in app.py
3. PDF report generation stability under various resolutions
"""

from pathlib import Path
import pytest
from PIL import Image
import io

from src.detection.category_checker import CategoryCompatibilityChecker
from app.utils.pdf_report import generate_inspection_pdf


def _get_sample_path(ds_rel: str, alt_rel: str) -> Path:
    p = Path(ds_rel)
    if p.exists():
        return p
    alt = Path(alt_rel)
    if alt.exists():
        return alt
    return p


def test_screw_model_with_zipper_image_mismatch():
    """Verify selecting screw model with a zipper image correctly detects category mismatch."""
    checker = CategoryCompatibilityChecker()
    zipper_path = _get_sample_path(
        "dataset/mvtec_anomaly_detection/zipper/test/good/001.png",
        "assets/test_samples/zipper/good_001.png"
    )
    if not zipper_path.exists():
        pytest.skip("Sample image not available")
        
    result = checker.check_compatibility(zipper_path, selected_category="screw")
    assert result["selected_category"] == "screw"
    assert result["is_mismatch"] is True
    assert result["best_compatible_category"] == "zipper"
    assert result["confidence_margin"] > 0.35


def test_bottle_model_with_transistor_image_mismatch():
    """Verify selecting bottle model with a transistor image correctly detects mismatch."""
    checker = CategoryCompatibilityChecker()
    transistor_path = _get_sample_path(
        "dataset/mvtec_anomaly_detection/transistor/test/good/001.png",
        "assets/test_samples/transistor/good_001.png"
    )
    if not transistor_path.exists():
        pytest.skip("Sample image not available")
        
    result = checker.check_compatibility(transistor_path, selected_category="bottle")
    assert result["selected_category"] == "bottle"
    assert result["is_mismatch"] is True
    assert result["best_compatible_category"] == "transistor"


def test_matching_category_no_mismatch():
    """Verify selecting screw model with a screw image does not flag a mismatch."""
    checker = CategoryCompatibilityChecker()
    screw_path = _get_sample_path(
        "dataset/mvtec_anomaly_detection/screw/test/good/001.png",
        "assets/test_samples/screw/good_001.png"
    )
    if not screw_path.exists():
        pytest.skip("Sample image not available")
        
    result = checker.check_compatibility(screw_path, selected_category="screw")
    assert result["selected_category"] == "screw"
    assert result["is_mismatch"] is False
    assert result["best_compatible_category"] == "screw"


def test_pdf_generation_high_res():
    """Verify PDF generates cleanly with high resolution images (e.g. 900x900)."""
    buf = io.BytesIO()
    im = Image.new("RGB", (900, 900), color=(120, 140, 160))
    im.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    inspection_data = {
        "status": "DEFECTIVE",
        "is_defective": True,
        "anomaly_score": 3.3602,
        "image_threshold": 2.8000,
        "decision_margin": 0.5602,
        "num_defects": 1,
        "localized_regions": [
            {
                "label": "Region 01",
                "bbox": [100, 150, 250, 300],
                "area": 22500,
                "score": 3.8,
                "intensity": "High"
            }
        ]
    }

    pdf_bytes = generate_inspection_pdf(
        inspection_data=inspection_data,
        category="bottle",
        filename="000.png",
        original_image_bytes=raw_bytes
    )
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 20000
    assert pdf_bytes.startswith(b"%PDF-")
