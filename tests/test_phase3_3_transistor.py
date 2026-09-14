"""
Phase 3.3 Transistor Targeted Improvement Tests
===============================================
Verifies:
1. Category configuration loading for Phase 3.3 transistor parameters.
2. PatchCoreDetectorV23 initialization and score method dispatch.
3. Protected category baseline preservation (bottle, leather, screw, zipper).
4. Dual-condition detection rule and border margin handling.
"""
import os
import sys
import json
from pathlib import Path
import pytest
import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_category_config
from src.detection.patchcore_v23_detector import PatchCoreDetectorV23


def test_transistor_phase33_config_loading():
    """Verify config.yaml correctly loads Phase 3.3 transistor parameters."""
    cfg = load_config()
    t_cfg = get_category_config("transistor", cfg)

    assert t_cfg["use_spatial_prior"] is False, "Transistor in Phase 3.3 must have spatial prior disabled"
    assert t_cfg["image_score_method"] == "top200", "Transistor must use top200 scoring"
    assert abs(t_cfg["image_threshold"] - 3.525) < 1e-3, "Transistor image threshold must be 3.525"
    assert abs(t_cfg["pixel_threshold"] - 2.822) < 1e-3, "Transistor pixel threshold must be 2.822"
    assert t_cfg["border_margin"] == 0, "Transistor border margin must be 0 to preserve boundary leads"
    assert t_cfg["min_region_area"] == 15, "Transistor min_region_area must be 15"
    assert t_cfg["morphology_kernel"] == 2, "Transistor morphology kernel must be 2"


def test_transistor_detector_initialization():
    """Verify PatchCoreDetectorV23 correctly initializes with Phase 3.3 parameters."""
    model_dir = Path("models/transistor/patchcore_v23")
    if not model_dir.exists():
        pytest.skip("Transistor model artifacts not found")

    device = torch.device("cpu")
    detector = PatchCoreDetectorV23(category="transistor", device=device)

    assert detector.category == "transistor"
    assert detector.use_spatial_prior is False
    assert detector.image_score_method == "top200"
    assert abs(detector.image_threshold - 3.525) < 1e-3
    assert abs(detector.pixel_threshold - 2.822) < 1e-3
    assert detector.border_margin == 0
    assert detector.min_region_area == 15
    assert detector.morphology_kernel == 2


def test_protected_categories_remain_untouched():
    """Verify bottle, leather, zipper, and screw configurations remain intact."""
    cfg = load_config()

    # Bottle protected baseline
    b_cfg = get_category_config("bottle", cfg)
    assert b_cfg["use_spatial_prior"] is True
    assert b_cfg["model_dir"] == "models/bottle/patchcore_v23"

    # Leather
    l_cfg = get_category_config("leather", cfg)
    assert l_cfg["use_spatial_prior"] is False

    # Screw
    s_cfg = get_category_config("screw", cfg)
    assert s_cfg["use_spatial_prior"] is True

    # Zipper
    z_cfg = get_category_config("zipper", cfg)
    assert z_cfg["use_spatial_prior"] is True


def test_phase33_results_file_exists():
    """Verify Phase 3.3 evaluation metrics JSON exists and has valid values."""
    metrics_path = Path("results/phase3/phase3_3_transistor/metrics.json")
    if not metrics_path.exists():
        pytest.skip("Phase 3.3 evaluation has not been executed yet")

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["category"] == "transistor"
    assert data["phase"] == "3.3"
    assert data["image_auroc"] >= 0.90, f"Expected AUROC >= 0.90, got {data['image_auroc']}"
    assert data["detection_rate"] >= 0.75, f"Expected Detection Rate >= 75%, got {data['detection_rate']}"
    assert data["normal_accuracy"] >= 0.90, f"Expected Normal Accuracy >= 90%, got {data['normal_accuracy']}"
    assert data["pixel_f1"] >= 0.30, f"Expected Pixel F1 >= 0.30, got {data['pixel_f1']}"
    assert data["detected_defective"] == 32
    assert data["normal_images"] == 60
    assert data["defective_images"] == 40
