# VisionInspect Project Cleanup & Audit Report

**Date:** September 25, 2026  
**Auditor:** Antigravity AI  
**Scope:** Complete project audit of local files, models, code dependencies, and runtime assets.

---

## 1. Executive Summary

A comprehensive dependency map was built across all Python files, configuration references, API endpoints, CLI entry points, and test suites. Every file proposed for deletion was audited to verify zero active imports, zero dynamic references, zero configuration references, and zero test dependencies.

Active models, datasets, test suites, inference pipelines, and frontend/backend servers were verified before and after cleanup to ensure zero regression.

---

## 2. Dependency & Reference Mapping

| Component | Active Implementation | Status |
|:---|:---|:---|
| **Frontend** | `app/app.py` (Streamlit standalone application) | **ACTIVE & PROTECTED** |
| **Backend API** | `app/backend.py` (FastAPI REST service) | **ACTIVE & PROTECTED** |
| **Active Inference Pipeline** | `src/detection/patchcore_v23_detector.py` | **ACTIVE & PROTECTED** |
| **Active PatchCore Architecture** | `src/models/patchcore_v22.py` (`PatchCoreModelV22`) | **ACTIVE & PROTECTED** |
| **CLI Tooling** | `inference.py`, `run_manual_inference.ps1` | **ACTIVE & PROTECTED** |
| **Configuration** | `config.yaml`, `src/utils/config.py` | **ACTIVE & PROTECTED** |
| **Data Pipelines** | `src/data/dataset_loader.py`, `src/data/preprocessing.py` | **ACTIVE & PROTECTED** |
| **Localization & Heatmaps** | `src/detection/localization.py`, `src/detection/heatmap.py` | **ACTIVE & PROTECTED** |
| **Evaluation Metrics** | `src/evaluation/metrics.py`, `src/evaluation/pixel_sweep.py`, `src/evaluation/roc_curve.py` | **ACTIVE & PROTECTED** |
| **Baseline Detectors** | `src/detection/anomaly_detector.py`, `src/detection/patchcore_detector.py`, `src/detection/patchcore_v22_detector.py` | **PRESERVED FOR TEST CONTRACTS** |

---

## 3. Files Identified as Obsolete & Confirmed for Deletion

The following files were confirmed to have zero active imports, zero CLI or test references, and are obsolete scratch or temporary files:

1. **`temp_pipeline.py`** (15 KB):
   - Legacy autoencoder scratch pipeline.
   - Explicitly listed in `.gitignore` under `# Scratch & temporary scripts`.
   - Zero active imports or dependencies.
2. **`test_env.py`** (461 bytes) & **`test_env.ipynb`** (618 bytes):
   - Initial environment smoke test scripts from early virtualenv setup.
   - Explicitly listed in `.gitignore` under `# Scratch & temporary scripts`.
   - Zero references in active test suites or applications.
3. **`desktop.ini`** (114 bytes):
   - OS-generated folder metadata file created by Windows Explorer.
   - Explicitly listed in `.gitignore` under `# OS and Editor specific`.
4. **`results/predictions/Old outputs/`** (45 files, ~11.3 MB):
   - Legacy image dumps from early Phase 1 and Phase 2 experiment iterations.
   - Explicitly ignored under `.gitignore` (`results/predictions/`).
   - Zero references in test suites, documentation, or frontend.
5. **Stale `*.cpython-311.pyc` bytecode** (27 files):
   - Stale Python 3.11 bytecode files located in `__pycache__` directories across `src/` and `tests/`.
   - Runtime is standardized on Python 3.12 (`visioninspect_py312`).
6. **Unused 0-byte Placeholder Files in `app/components/` and `app/styles/`**:
   - `app/components/batch_results.py` (0 bytes)
   - `app/components/heatmap_viewer.py` (0 bytes)
   - `app/components/inspection.py` (0 bytes)
   - `app/components/reports.py` (0 bytes)
   - `app/components/upload.py` (0 bytes)
   - `app/styles/style.css` (0 bytes)
   - Created during initial project setup on 2026-08-26, never populated, never imported.

---

## 4. Files Intentionally Preserved

The following files were reviewed and intentionally preserved:

1. **`setup_project_structure.py`**:
   - Project repository scaffolding definition; preserved as architecture documentation.
2. **`PHASE_3_PRE_IMPLEMENTATION_AUDIT.md`**:
   - Comprehensive system audit and benchmark specification; preserved for viva/examination documentation.
3. **`notebooks/`**:
   - `01_dataset_exploration.ipynb` (354 KB)
   - `03_autoencoder_training.ipynb` (5.5 KB)
   - `04_anomaly_detection.ipynb` (3.38 MB)
   - `plot1.png`, `plot2.png`
   - Preserved for interactive data science exploration and viva review.
4. **`scripts/`** (All 24 utility scripts):
   - `train_patchcore.py`, `train_patchcore_v22.py`
   - `calibrate_and_lock_thresholds.py`, `calibrate_phase3_2.py`
   - `run_phase3_2_eval.py`, `run_phase3_3_eval.py`, `run_multi_category_eval.py`
   - `diagnose_localization_pipeline.py`, `diagnose_phase2_3.py`, `recompute_spatial_prior.py`
   - All evaluation and calibration scripts are preserved to ensure 100% reproducibility of trained models.
5. **`docs/`**:
   - All technical reports and documentation preserved.
6. **`assets/`**:
   - `carton_box_clean.png`, `carton_box_scan.png`, `carton_box_scan.jpg`: Active UI hero visuals on Home page.

---

## 5. Model Files Preserved (100% Retained)

Every single model directory and weight file was verified and retained:

- **`models/bottle/patchcore_v23/`** (`memory_bank.pt`, `metadata.json`, `spatial_prior.pt`, `spatial_prior_p75.pt`)
- **`models/leather/patchcore_v23/`** (`memory_bank.pt`, `metadata.json`, `spatial_prior.pt`)
- **`models/transistor/patchcore_v23/`** (`memory_bank.pt`, `metadata.json`, `spatial_prior.pt`)
- **`models/zipper/patchcore_v23/`** (`memory_bank.pt`, `metadata.json`, `spatial_prior.pt`)
- **`models/screw/patchcore_v23/`** (`memory_bank.pt`, `metadata.json`, `spatial_prior.pt`)
- **`models/bottle/patchcore_v22/`** (Retained for V2.2 regression testing)
- **`models/bottle/patchcore/`** (Retained for V2.1 regression testing)
- **`models/bottle/autoencoder.pth`** (Retained for baseline testing)
- **`models/baseline/bottle/`** (Retained for baseline comparison)
- **`models/category_prototypes.pt`** (Retained for zero-shot prototype comparisons)

---

## 6. Datasets Preserved (100% Retained)

The full MVTec Anomaly Detection dataset for all industrial categories is strictly protected under `dataset/mvtec_anomaly_detection/`. No dataset images, ground-truth masks, or splits were modified or deleted.

---

## 7. Tests Preserved (100% Retained)

All 13 test suites are preserved and active:
- `tests/test_inspect_page.py`
- `tests/test_frontend_state.py`
- `tests/test_app.py`
- `tests/test_dataset.py`
- `tests/test_detection.py`
- `tests/test_model.py`
- `tests/test_model_dispatch.py`
- `tests/test_patchcore.py`
- `tests/test_patchcore_v22.py`
- `tests/test_patchcore_v23.py`
- `tests/test_phase3_1_integrity.py`
- `tests/test_phase3_2_batch_compatibility.py`
- `tests/test_phase3_2_localization.py`
- `tests/test_phase3_3_transistor.py`
- `tests/test_phase3_multi_category.py`

---

## 8. Review Required (Not Deleted)

1. **`venv/`** (Alternative Python 3.14 virtual environment):
   - The active project environment is `visioninspect_py312` (Python 3.12).
   - `venv/` is an inactive Python 3.14 environment.
   - Retained per the principle: Never delete virtual environments without explicit user instruction.
2. **`notebooks/02_data_preprocessing.ipynb` & `notebooks/05_evaluation.ipynb`** (0 bytes):
   - Empty placeholder notebooks; retained in case the user intended to develop additional notebook workflows.
