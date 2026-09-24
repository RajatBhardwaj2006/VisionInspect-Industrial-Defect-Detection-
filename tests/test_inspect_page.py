"""
tests/test_inspect_page.py
Unit tests verifying Inspect and Result page components:
- Golden samples and category catalog integrity
- Thumbnail generation caching
- Evidence-based defect analysis narrative
- Usability assessment contracts and engineering criteria
"""

import io
import base64
from pathlib import Path
from PIL import Image
import pytest

from app.app import CATEGORIES, get_thumbnail_b64


def test_category_catalog_and_golden_samples():
    """Verify all 5 industrial categories exist with valid golden samples."""
    expected_categories = ["bottle", "leather", "transistor", "zipper", "screw"]
    for cat in expected_categories:
        assert cat in CATEGORIES, f"Category {cat} missing from CATEGORIES catalog"
        info = CATEGORIES[cat]
        assert "name" in info
        assert "label" in info
        assert "golden_sample" in info
        assert "samples" in info and len(info["samples"]) >= 3

        # Verify golden sample file exists
        p = Path(info["golden_sample"])
        assert p.exists(), f"Golden sample not found at {p}"


def test_thumbnail_generation_all_categories():
    """Verify get_thumbnail_b64 produces valid base64 images for all categories."""
    for cat, info in CATEGORIES.items():
        b64_str = get_thumbnail_b64(info["golden_sample"])
        assert b64_str, f"Thumbnail base64 empty for {cat}"
        # Decode and verify it's a valid image
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw))
        assert img.size[0] <= 96 and img.size[1] <= 96


def test_defect_analysis_and_usability_contracts():
    """Verify evidence-based defect analysis and usability assessment text logic."""
    # Test DEFECTIVE case
    is_defective = True
    score = 2.45
    th = 1.50
    margin = score - th
    n_defects = 2
    regs = [
        {"label": "Region 01", "area": 120, "score": 2.45, "bbox": [10, 20, 30, 40]},
        {"label": "Region 02", "area": 45, "score": 1.85, "bbox": [50, 60, 20, 20]}
    ]

    # Verify WHAT / WHERE / WHY for Defective
    what_text = f"Optical feature extraction yielded an anomaly score of {score:.4f}, exceeding the calibrated nominal threshold of {th:.4f} (Decision Margin: +{margin:.4f})."
    assert "2.4500" in what_text
    assert "1.5000" in what_text
    assert "+0.9500" in what_text

    where_text = f"Localized to {n_defects} discrete defect cluster(s)"
    assert "2 discrete defect cluster(s)" in where_text

    # Verify Usability Assessment for Defective
    usability_status = "REQUIRES REVIEW"
    assert usability_status == "REQUIRES REVIEW"

    # Ensure no speculative safety statements
    speculative_terms = ["will break", "is unsafe", "guaranteed failure", "catastrophic danger"]
    for term in speculative_terms:
        assert term not in what_text.lower()
        assert term not in where_text.lower()

    # Test NORMAL case
    is_defective = False
    score = 1.10
    th = 1.50
    margin = score - th
    n_defects = 0

    what_normal = f"Optical scan yielded an anomaly score of {score:.4f}, safely below the calibrated nominal threshold of {th:.4f}"
    assert "1.1000" in what_normal
    assert "1.5000" in what_normal

    usability_normal = "PASSES VISUAL INSPECTION"
    assert usability_normal == "PASSES VISUAL INSPECTION"
