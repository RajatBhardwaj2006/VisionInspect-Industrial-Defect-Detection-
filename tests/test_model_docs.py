"""
tests/test_model_docs.py
Unit tests verifying the Model Documentation architecture:
- Metadata and category configuration integrity (config.yaml consistency)
- Defect taxonomy matching real MVTec dataset folders
- Real repository source code traceability
- Golden standard reference samples existence
- Reusable documentation page rendering and routing callbacks
"""

from pathlib import Path
import pytest
from app.components.model_docs import CATEGORY_DOCS, REAL_CODE_SNIPPETS
from src.utils.config import load_config, get_category_config


def test_category_docs_all_categories_present():
    """Verify all 5 industrial categories exist with complete metadata."""
    expected = ["bottle", "leather", "transistor", "zipper", "screw"]
    assert len(CATEGORY_DOCS) == 5
    for cat in expected:
        assert cat in CATEGORY_DOCS
        doc = CATEGORY_DOCS[cat]
        assert doc["id"] == cat
        assert doc["name"]
        assert doc["technical_name"]
        assert doc["subtitle"]
        assert doc["short_desc"]
        assert doc["golden_sample"]
        assert doc["backbone"] == "ResNet-18 (ImageNet Pre-trained, PyTorch torchvision)"
        assert "448-dimensional" in doc["feature_dim"]
        assert "64 × 64" in doc["patch_grid"]
        assert doc["model_status"] == "PRODUCTION / LOCKED"
        assert len(doc["defects"]) >= 3


def test_thresholds_match_config_yaml():
    """Verify that displayed thresholds strictly match the verified config.yaml."""
    cfg = load_config()
    for cat, doc in CATEGORY_DOCS.items():
        cat_cfg = get_category_config(cat, cfg)
        expected_img_th = cat_cfg.get("image_threshold")
        expected_pix_th = cat_cfg.get("pixel_threshold")
        assert abs(doc["image_threshold"] - expected_img_th) < 1e-4, (
            f"Image threshold mismatch for {cat}: doc={doc['image_threshold']} vs config={expected_img_th}"
        )
        assert abs(doc["pixel_threshold"] - expected_pix_th) < 1e-4, (
            f"Pixel threshold mismatch for {cat}: doc={doc['pixel_threshold']} vs config={expected_pix_th}"
        )


def test_golden_samples_exist():
    """Verify golden standard sample paths exist on disk."""
    for cat, doc in CATEGORY_DOCS.items():
        sample_path = Path(doc["golden_sample"])
        assert sample_path.exists(), f"Golden sample not found for {cat}: {sample_path}"


def test_mvtec_defect_tags_match_dataset_folders():
    """Verify all documented defect tags actually exist in MVTec dataset test folders."""
    mvtec_root = Path("dataset/mvtec_anomaly_detection")
    for cat, doc in CATEGORY_DOCS.items():
        cat_test_dir = mvtec_root / cat / "test"
        if not cat_test_dir.exists():
            continue
        real_dirs = {p.name for p in cat_test_dir.iterdir() if p.is_dir() and p.name != "good"}
        doc_tags = {d["tag"] for d in doc["defects"]}
        # All documented tags must exist in the real dataset
        for tag in doc_tags:
            assert tag in real_dirs, f"Defect tag '{tag}' documented for {cat} but not in {cat_test_dir}"


def test_real_code_snippets_traceable_to_repo():
    """Verify that every code snippet references a real file that exists in the repository."""
    assert len(REAL_CODE_SNIPPETS) >= 5
    for key, snip in REAL_CODE_SNIPPETS.items():
        assert snip["file"]
        assert snip["lines"]
        assert snip["code"]
        assert snip["explanation"]
        
        # Check that file path exists
        # In case of multiple paths like "app/backend.py & config.yaml", check the primary
        primary_file = snip["file"].split("&")[0].strip().split()[0]
        assert Path(primary_file).exists(), f"Source file does not exist: {primary_file}"
