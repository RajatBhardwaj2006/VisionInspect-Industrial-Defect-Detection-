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

from app.app import CATEGORIES, get_thumbnail_b64, get_simple_explanation


def test_category_catalog_and_golden_samples():
    """Verify strictly all 5 industrial categories exist with valid golden samples and no carton model."""
    expected_categories = ["bottle", "leather", "transistor", "zipper", "screw"]
    assert len(CATEGORIES) == 5, f"Expected exactly 5 categories, found {len(CATEGORIES)}"
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

    assert "carton" not in CATEGORIES
    assert "carton_box" not in CATEGORIES
    assert "box" not in CATEGORIES


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
    """Verify evidence-based defect analysis and plain-English usability assessment text logic."""
    jargon_blacklist = ["448-dimensional", "resnet", "manifold", "coreset", "euclidean"]

    # 1. Defective Screw with thread defect
    what, where, why = get_simple_explanation("screw", "thread_side/001.png", is_defective=True, score=0.62, threshold=0.45, num_defects=1)
    assert "screw thread" in what
    assert "1 localized region was found near the screw thread." in where
    assert "cross the inspection threshold" in why
    for term in jargon_blacklist:
        assert term not in what.lower()
        assert term not in where.lower()
        assert term not in why.lower()

    # 2. Defective Bottle with mouth crack
    what, where, why = get_simple_explanation("bottle", "broken_large/002.png", is_defective=True, score=0.75, threshold=0.35, num_defects=2)
    assert "bottle opening" in what
    assert "2 localized regions were found around the bottle rim." in where
    assert "cross the inspection threshold" in why

    # 3. Defective Transistor with bent pin
    what, where, why = get_simple_explanation("transistor", "bent_lead/003.png", is_defective=True, score=0.58, threshold=0.40, num_defects=1)
    assert "transistor pins" in what
    assert "near the connection pins" in where

    # 4. Defective Zipper with broken teeth
    what, where, why = get_simple_explanation("zipper", "broken_teeth/004.png", is_defective=True, score=0.82, threshold=0.50, num_defects=1)
    assert "zipper" in what
    assert "along the zipper teeth" in where

    # 5. Defective Leather
    what, where, why = get_simple_explanation("leather", "cut/005.png", is_defective=True, score=0.91, threshold=0.42, num_defects=3)
    assert "leather surface or texture" in what
    assert "3 localized regions were found on the leather surface." in where

    # 6. Normal Sample
    what_n, where_n, why_n = get_simple_explanation("screw", "good/000.png", is_defective=False, score=0.32, threshold=0.45, num_defects=0)
    assert what_n == "No unusual areas were detected."
    assert "expected normal pattern" in where_n
    assert "stayed below the anomaly threshold" in why_n
    for term in jargon_blacklist:
        assert term not in what_n.lower()
        assert term not in where_n.lower()
        assert term not in why_n.lower()

    # 7. Usability Assessment copy validation
    usability_defective = "VisionInspect found a visual difference that needs to be reviewed. The system detects visual anomalies; it does not independently certify whether a physical component is safe to use."
    usability_normal = "No significant visual difference was detected under the current inspection criteria."
    assert "does not independently certify" in usability_defective
    assert "No significant visual difference was detected" in usability_normal
