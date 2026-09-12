# VisionInspect Phase 3.0: Multi-Category Industrial Defect Detection Framework

**Date:** September 2026  
**Maintainer:** [Rajat Bhardwaj](https://github.com/RajatBhardwaj2006)  
**Repository:** [VisionInspect-Industrial-Defect-Detection-](https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git)

---

## 1. Executive Summary

Phase 3.0 marks the transition of VisionInspect from a single-product prototype (bottle) into an extensible, multi-category industrial visual inspection framework. Built around the proven high-resolution PatchCore v2.3 architecture (ResNet18 backbone, 64x64 feature grid, 448-dimensional patch representations), Phase 3.0 introduces:

1. **Multi-Category Architecture**: Unified training and evaluation pipeline scaling across both rigid geometric objects and unaligned textural surfaces.
2. **Initial 5 Categories**:
   - `bottle` (protected baseline)
   - `leather` (unaligned surface texture)
   - `transistor` (electronic component)
   - `zipper` (periodic structural object)
   - `screw` (fine-scale metal hardware)
3. **Adaptive Spatial Prior Subtraction**: Configurable per category to eliminate normal background edges without distorting unaligned textures (`use_spatial_prior: false` for `leather`).
4. **Empirical 2D Threshold Calibration**: Category-specific optimal calibration for Image Anomaly Score and Pixel Segmentation Thresholds.
5. **Production Inspection Stack**:
   - **FastAPI Backend (`app/backend.py`)**: Endpoints for `/health`, `/categories`, and `/predict`.
   - **Streamlit Web UI (`app/app.py`)**: Real-time visual inspection with bounding box overlays, anomaly heatmaps, and defect metric tables.
6. **Exploratory Data Analysis (EDA)**: Complete characterization of dataset distributions and defect area percentages across all 5 categories.

---

## 2. Dataset Distribution & EDA Findings

The MVTec Anomaly Detection dataset exhibits substantial diversity in surface geometry, defect scale, and defect types:

| Category | Type | Train Normal | Test Normal | Test Defective | Total Test | Defect Classes | Resolution | Mean Defect Area | Max Defect Area |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** | Object | 209 | 20 | 63 | 83 | 3 | 900x900 | 7.62% | 27.42% |
| **leather** | Texture | 245 | 32 | 92 | 124 | 5 | 1024x1024 | 0.87% | 6.86% |
| **transistor** | Object | 213 | 60 | 40 | 100 | 4 | 1024x1024 | 11.98% | 49.32% |
| **zipper** | Object | 240 | 32 | 119 | 151 | 7 | 1024x1024 | 2.62% | 11.83% |
| **screw** | Object | 320 | 41 | 119 | 160 | 5 | 1024x1024 | 0.34% | 1.01% |

### Key Observations:
- **Defect Scale Disparity**: Defect areas range from microscopic thread scratches on screws (mean 0.34% of image area, ~3,518 pixels) to massive broken glass sections on bottles (up to 27.42%) and damaged transistor casings (up to 49.32%).
- **Texture vs. Geometric Invariance**: Textures like `leather` have no static spatial alignment; normal spatial priors subtract artificial patterns. Disabling the spatial prior on `leather` resulted in 100% normal image accuracy (0 false positives).

---

## 3. Verified Performance Benchmark (5 Categories)

Each category was trained on normal samples only, followed by empirical 2D threshold grid sweeps on the test sets to maximize pixel F1-score:

| Category | Image AUROC | Pixel Precision | Pixel Recall | Pixel F1 | Pixel IoU | Defect Detection Rate | Normal Image Accuracy | Image Threshold | Pixel Threshold | Spatial Prior |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** *(Protected)* | **0.9929** | **0.7079** | **0.6915** | **0.6996** | **0.5380** | 95.2% (60/63) | 70.0% (14/20) | 1.50 | 1.40 | True |
| **leather** | **0.9236** | **0.5366** | **0.5249** | **0.5307** | **0.3612** | 72.8% (67/92) | **100.0%** (32/32) | 2.20 | 2.71 | False |
| **zipper** | **0.8981** | **0.5152** | **0.6460** | **0.5732** | **0.4018** | 84.0% (100/119) | 62.5% (20/32) | 1.47 | 0.91 | True |
| **screw** | **0.8776** | **0.4175** | **0.5885** | **0.4884** | **0.3231** | 84.9% (101/119) | 65.9% (27/41) | 2.28 | 2.00 | True |
| **transistor** | **0.8200** | **0.4354** | **0.3898** | **0.4113** | **0.2589** | 62.5% (25/40) | 76.7% (46/60) | 2.69 | 0.88 | True |

> **Baseline Integrity:** The Phase 2.3 bottle benchmark (`AUROC=0.9929`, `F1=0.6996`, `IoU=0.5380`) remains 100% intact and unchanged.

---

## 4. Software Architecture & API Endpoints

### 4.1 Category-Aware Detector (`src/detection/patchcore_v23_detector.py`)
- Dynamic resolution of model directories: `models/<category>/patchcore_v23`.
- Validates category against model `metadata.json` to prevent accidental category mismatch.
- Dual input support: accepts filesystem path strings or in-memory `PIL.Image` instances.

### 4.2 FastAPI Inspection Service (`app/backend.py`)
- `GET /health`: Returns service status, active device (CUDA/CPU), supported categories, and loaded models.
- `GET /categories`: Returns metadata for all supported categories, including calibrated thresholds, backbone info, and training status.
- `POST /predict`: Performs end-to-end defect detection on uploaded images, returning classification status, calibrated thresholds, defect bounding boxes, and base64-encoded visual overlays.

### 4.3 Streamlit Interactive Dashboard (`app/app.py`)
- Category selector with live parameter display.
- Dual mode: sample browser directly querying the MVTec test sets or drag-and-drop custom image upload.
- Full inspection view: normal/defective decision banners, anomaly score delta, heatmap overlay, and localized defect region tables.

---

## 5. Verification & Test Suite

The test suite validates:
- Core dataset loaders and preprocessing pipelines.
- Autoencoder baseline models.
- PatchCore v2.2 and v2.3 feature extractors, spatial prior computation, and coreset memory banks.
- Multi-category detector initialization, threshold enforcement, and category mismatch rejection.
- FastAPI backend endpoints (`/health`, `/categories`, `/predict` error handling and inference).

Run tests:
```powershell
python -m pytest tests/ -v
```

---

## 6. Phase 3.1 Evaluation Integrity & Workflow Update

In Phase 3.1, the evaluation methodology was upgraded to resolve test-set threshold calibration leakage:
- **Historical Benchmark Preservation:** Historical test-calibrated metrics are preserved in `results/phase3/calibration_test_set/`.
- **Unsupervised Normal Calibration:** Decision thresholds are calibrated strictly on held-out normal samples (`train/good/`) without test set access.
- **Unbiased Final Test:** Evaluated with locked parameters in `results/phase3/final_test/`.
- See [`docs/phase3_evaluation_integrity.md`](phase3_evaluation_integrity.md) for full methodology and unbiased benchmarks.

---

## 7. Phase 3.2 Batch Inspection & Category Compatibility Prevention

Phase 3.2 expands the production operational layer with batch processing capabilities and configuration guardrails:

1. **Batch Upload & Independent Processing:**
   - Supports uploading and inspecting batches of 1 to 5 images simultaneously.
   - Rejects batches exceeding 5 images with clear feedback.
   - Independent per-image error resilience: corrupted files report `status: ERROR` without interrupting the evaluation of remaining valid images.
2. **Category Compatibility Checking:**
   - ResNet18 PatchCore 448-dimensional feature representations are checked against precomputed normal category prototypes (`models/category_prototypes.pt`).
   - Warns operators when an uploaded image is markedly closer to another category (e.g., uploading a `zipper` when `bottle` is selected) with confidence margin $\ge 0.35$.
   - Strictly conservative: does not flag false alarms on genuine defective items of the active category.
   - **User Autonomy Principle:** Never automatically switches categories; presents actionable advice with explicit operator controls (`[Switch to <Suggested>]` and `[Continue with <Selected>]`).
3. **Operational Transparency & UI State Management:**
   - Real-time model change notifications (`MODEL CHANGED: Bottle → Zipper`).
   - Displays safe logical model paths (`models/<category>/patchcore_v23/`) with zero local filesystem disclosure.
   - Comprehensive batch summary analytics: Total, Normal, Defective, Category Warnings, and Failed counts.
   - Robust session state clearing upon category changes or new batch file selections.

---

## 8. Category-Specific Observations & Future Research: Transistor Analysis

During the Phase 3 evaluation benchmarking, the `transistor` category exhibited an asymmetric performance profile:
- **Normal-Image Specificity:** **100.0%** (zero false alarms on normal samples in unsupervised calibration, 76.7% in full test).
- **Defect Detection Rate:** **62.5%** (25/40 defective samples detected).
- **Pixel-Level Recall:** **10.6%** (unbiased test set recall).

### Root-Cause Diagnosis:
1. **Fine-Scale Geometry vs. Feature Receptive Field:**
   - Defect classes in the transistor category include microscopic bent pins, damaged leads, and cut wires.
   - PatchCore v2.3 utilizes a composite multi-scale embedding combining ResNet18 Layer 1, Layer 2, and Layer 3. The Layer 3 receptive field is expansive and tailored for macro-surface textures (like leather) or broad structural bodies (like bottles).
   - Fine pin displacements produce subtle local spatial shifts that get diluted in 448-dimensional feature pooling.
2. **Alignment Sensitivity:**
   - Transistor orientation and lead placements require precise spatial keypoint registration. Spatial background subtraction partially compensates for static housing features but cannot easily isolate minute lead deformations.

### Prescribed Protocol:
- **No Retraining or Premature Architecture Replacement in Phase 3:** In accordance with project integrity principles, the transistor model and thresholds are locked and not artificially tuned or overwritten.
- **Future Research Directions:**
  - Layer 1-dominated feature concatenation for sub-millimeter electronic components.
  - High-resolution localized patch grids (128x128).
  - Keypoint-guided pin alignment preprocessing for micro-electronics.


