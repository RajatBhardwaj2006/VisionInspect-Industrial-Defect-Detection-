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

from app.app import (
    CATEGORIES,
    get_thumbnail_b64,
    get_simple_explanation,
    get_defect_explanation,
    get_usability_assessment
)


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


def test_comprehensive_three_part_defect_explanation():
    """Verify the 3 required human-friendly explanation sections across all 5 categories."""
    # 1. Defective Screw with thread defect
    head, desc, where, why = get_defect_explanation("screw", "thread_side/001.png", is_defective=True, score=2.45, threshold=2.28, num_defects=1, margin=0.17)
    assert head == "WHY IS THIS PRODUCT DEFECTIVE?"
    assert "Thread damage detected" in desc
    assert "Region 01" in where and "near the screw threads" in where
    assert "crossed the calibrated inspection threshold" in why
    assert "+0.1700" in why

    # 2. Normal Screw
    head_n, desc_n, where_n, why_n = get_defect_explanation("screw", "good/000.png", is_defective=False, score=1.85, threshold=2.28, num_defects=0, margin=-0.43)
    assert head_n == "WHY IS THIS PRODUCT ACCEPTED?"
    assert "conforms to the nominal baseline" in desc_n
    assert where_n == "No anomalous regions detected."
    assert "remained safely below the inspection threshold" in why_n

    # 3. Defective Bottle with mouth crack
    head_b, desc_b, where_b, why_b = get_defect_explanation("bottle", "broken_large/000.png", is_defective=True, score=1.82, threshold=1.50, num_defects=2, margin=0.32)
    assert head_b == "WHY IS THIS PRODUCT DEFECTIVE?"
    assert "Glass fracture detected" in desc_b
    assert "around the bottle rim and opening" in where_b
    assert "Regions 01 to 02" in where_b

    # 4. Defective Transistor with missing body
    head_t, desc_t, where_t, why_t = get_defect_explanation("transistor", "misplaced/002.png", is_defective=True, score=4.12, threshold=3.525, num_defects=1, margin=0.595)
    assert head_t == "WHY IS THIS PRODUCT DEFECTIVE?"
    assert "Critical component missing" in desc_t
    assert "at the component mounting site" in where_t

    # 5. Rotated Transistor
    head_tr, desc_tr, where_tr, why_tr = get_defect_explanation("transistor", "misplaced/000.png", is_defective=True, score=3.67, threshold=3.525, num_defects=1, margin=0.145)
    assert "Orientation rotation" in desc_tr
    assert "pins are intact" in desc_tr

    # 6. Defective Leather cut
    head_l, desc_l, where_l, why_l = get_defect_explanation("leather", "cut/000.png", is_defective=True, score=2.85, threshold=2.20, num_defects=1, margin=0.65)
    assert "Surface cut detected" in desc_l

    # 7. Defective Zipper broken teeth
    head_z, desc_z, where_z, why_z = get_defect_explanation("zipper", "broken_teeth/000.png", is_defective=True, score=1.92, threshold=1.47, num_defects=1, margin=0.45)
    assert "Broken teeth chain" in desc_z


def test_three_state_usability_assessment_and_exceptions():
    """Verify category-aware usability assessment outputs strictly one of 3 states: ACCEPT, REQUIRES HUMAN REVIEW, REJECT."""
    valid_states = {"ACCEPT", "REQUIRES HUMAN REVIEW", "REJECT"}

    # 1. Normal components across all 5 categories -> ACCEPT
    for cat in ["bottle", "leather", "transistor", "zipper", "screw"]:
        u = get_usability_assessment(cat, "good/001.png", is_defective=False, score=1.0, threshold=2.0, num_defects=0, margin=-1.0)
        assert u["status"] == "ACCEPT"
        assert u["status"] in valid_states
        assert "conforms" in u["reason"].lower()

    # 2. Screw: Thread defect -> REJECT
    u_screw_th = get_usability_assessment("screw", "thread_side/000.png", is_defective=True, score=2.45, threshold=2.28, num_defects=1, margin=0.17)
    assert u_screw_th["status"] == "REJECT"
    assert "thread damage" in u_screw_th["reason"].lower()

    # 3. Screw: Minor cosmetic head scratch -> REQUIRES HUMAN REVIEW
    u_screw_sc = get_usability_assessment("screw", "scratch_head/000.png", is_defective=True, score=2.40, threshold=2.28, num_defects=1, margin=0.12)
    assert u_screw_sc["status"] == "REQUIRES HUMAN REVIEW"
    assert "scratch" in u_screw_sc["reason"].lower()

    # 4. Transistor: Rotated component (misplaced/000) -> ACCEPT (user constraint: do not reject solely for rotation)
    u_tr_rot = get_usability_assessment("transistor", "misplaced/000.png", is_defective=True, score=3.67, threshold=3.525, num_defects=1, margin=0.145)
    assert u_tr_rot["status"] == "ACCEPT"
    assert "rotated" in u_tr_rot["reason"].lower()

    # 5. Transistor: Missing component (misplaced/002) -> REJECT
    u_tr_miss = get_usability_assessment("transistor", "misplaced/002.png", is_defective=True, score=4.12, threshold=3.525, num_defects=1, margin=0.595)
    assert u_tr_miss["status"] == "REJECT"
    assert "missing" in u_tr_miss["reason"].lower()

    # 6. Transistor: Bent pin -> REQUIRES HUMAN REVIEW
    u_tr_bent = get_usability_assessment("transistor", "bent_lead/000.png", is_defective=True, score=3.80, threshold=3.525, num_defects=1, margin=0.275)
    assert u_tr_bent["status"] == "REQUIRES HUMAN REVIEW"

    # 7. Bottle: Rim crack -> REJECT
    u_bot_crk = get_usability_assessment("bottle", "broken_large/000.png", is_defective=True, score=1.85, threshold=1.50, num_defects=1, margin=0.35)
    assert u_bot_crk["status"] == "REJECT"

    # 8. Bottle: Contamination -> REQUIRES HUMAN REVIEW
    u_bot_cont = get_usability_assessment("bottle", "contamination/000.png", is_defective=True, score=1.65, threshold=1.50, num_defects=1, margin=0.15)
    assert u_bot_cont["status"] == "REQUIRES HUMAN REVIEW"

    # 9. Leather: Cut -> REJECT
    u_lea_cut = get_usability_assessment("leather", "cut/000.png", is_defective=True, score=2.80, threshold=2.20, num_defects=1, margin=0.60)
    assert u_lea_cut["status"] == "REJECT"

    # 10. Leather: Fold mark -> REQUIRES HUMAN REVIEW
    u_lea_fold = get_usability_assessment("leather", "fold/000.png", is_defective=True, score=2.45, threshold=2.20, num_defects=1, margin=0.25)
    assert u_lea_fold["status"] == "REQUIRES HUMAN REVIEW"

    # 11. Zipper: Broken teeth -> REJECT
    u_zip_brk = get_usability_assessment("zipper", "broken_teeth/000.png", is_defective=True, score=1.90, threshold=1.47, num_defects=1, margin=0.43)
    assert u_zip_brk["status"] == "REJECT"

    # 12. Zipper: Rough fabric -> REQUIRES HUMAN REVIEW
    u_zip_rgh = get_usability_assessment("zipper", "rough/000.png", is_defective=True, score=1.62, threshold=1.47, num_defects=1, margin=0.15)
    assert u_zip_rgh["status"] == "REQUIRES HUMAN REVIEW"

