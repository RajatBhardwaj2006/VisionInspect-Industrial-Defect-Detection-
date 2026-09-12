# VisionInspect Phase 3.1: Evaluation Integrity & Real Inspection Workflow

**Date:** September 2026  
**Maintainer:** [Rajat Bhardwaj](https://github.com/RajatBhardwaj2006)  
**Repository:** [VisionInspect-Industrial-Defect-Detection-](https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git)

---

## 1. Executive Summary

Phase 3.1 addresses two critical architectural requirements:
1. **Evaluation Scientific Integrity (Goal B)**: Resolves the methodological flaw where Phase 3.0 calibrated decision thresholds directly against the final MVTec test set ground-truth annotations (data leakage). We implement a strict **Train → Normal Validation Calibration → Final Locked Test** protocol without peeking at test labels.
2. **Generic Real-Image Inspection Workflow (Goal A)**: Eliminates confusing sample-by-default behavior in the Streamlit UI, adds primary drag-and-drop and file browsing for user-uploaded custom images (`PNG`, `JPG`, `JPEG`, `WEBP`), introduces strict state isolation (no stale predictions across uploads or category switches), and enriches FastAPI with root entrypoint (`GET /`) and interactive Swagger documentation (`/docs`).

---

## 2. Evaluation Integrity: Methodology & Analysis

### 2.1 The Problem with Test-Set Threshold Calibration
In many machine learning benchmarks, researchers inadvertently compute 2D grid sweeps over test images to select the "optimal" image and pixel thresholds that maximize test F1-score. While this identifies the upper theoretical ceiling of a model architecture, **it constitutes data leakage**. In production manufacturing:
- Ground-truth defect masks do not exist beforehand.
- The model must decide whether a product is defective using decision thresholds calibrated strictly before deployment.
- Evaluating a model on the same data used to select its hyperparameters produces over-optimistic results that cannot generalize to unseen production runs.

### 2.2 Unsupervised Normal Validation Strategy
Standard MVTec AD does not include an official labeled validation split. It provides only normal training samples (`train/good/`) and test samples (`test/<defect_types>/`).
To avoid inventing fictitious labels or altering scientific assumptions:
1. We construct a held-out **20% normal calibration split** from `train/good/` for each category.
2. We extract patch features and compute anomaly maps against the category coreset memory bank using ResNet18 (layers 1+2+3, 64x64 grid).
3. We empirically characterize the baseline anomaly score distribution on clean products:
   - Normal score mean ($\mu$) and standard deviation ($\sigma$).
   - 95th, 98th, and 99th percentiles of normal image scores.
   - 99.5th percentile of normal surface pixel scores.
4. **Threshold Selection Rules**:
   - **Image Threshold ($T_{image}$)**: Calibrated to achieve $\ge 98\%$ specificity on normal validation images (bounding false alarm rate $\le 2\%$).
   - **Pixel Threshold ($T_{pixel}$)**: Set at the 99.5th percentile of normal surface pixel noise. Any localized patch exceeding this noise level is subject to morphological filtering and connected component extraction.
5. **Parameter Locking**:
   All hyperparameters ($T_{image}$, $T_{pixel}$, $\sigma$, morphology kernel, min region area, spatial prior flag) are saved into `results/phase3/final_validation_locked/locked_thresholds.json` and locked.
6. **Zero-Leakage Final Test**:
   The final MVTec test set is evaluated strictly using the locked parameters with zero tuning or sweeps.

---

## 3. Validation Calibration & Locked Hyperparameters

Obtained by running `scripts/calibrate_and_lock_thresholds.py` on held-out normal calibration splits:

| Category | Normal Train Total | Held-out Val Normal | Normal Score Mean ± Std | Normal P95 | Normal P98 | Normal Pixel P99.5 | Locked $T_{image}$ | Locked $T_{pixel}$ | Spatial Prior |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** | 209 | 41 | 1.3627 ± 0.3183 | 1.9423 | 2.0226 | 0.6621 | **2.06** | **0.66** | Enabled |
| **leather** | 245 | 49 | 2.1240 ± 0.2911 | 2.6638 | 2.8983 | 1.7129 | **2.96** | **1.71** | **Disabled** |
| **transistor** | 213 | 42 | 2.0504 ± 0.5609 | 2.9298 | 3.3460 | 1.1785 | **3.41** | **1.18** | Enabled |
| **zipper** | 240 | 48 | 1.1198 ± 0.5877 | 1.8781 | 2.6624 | 0.5684 | **2.72** | **0.57** | Enabled |
| **screw** | 320 | 64 | 2.2073 ± 0.2382 | 2.5581 | 2.8914 | 1.6560 | **2.95** | **1.66** | Enabled |

*Artifacts saved to:* `results/phase3/final_validation_locked/locked_thresholds.json` *and* `results/phase3/final_validation_locked/validation_summary.csv`

---

## 4. Benchmark Comparison: Historical Test-Calibrated vs. Unbiased Final Locked

### 4.1 Historical Test-Set Calibrated Results (Phase 3.0)
*Preserved in:* `results/phase3/calibration_test_set/`
> **Methodology Note:** Optimized thresholds directly on the test set ground truth to identify peak upper-bound capability.

| Category | Image AUROC | Pixel Precision | Pixel Recall | Pixel F1 | Pixel IoU | Detection Rate | Normal Accuracy | Image Thresh | Pixel Thresh |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** *(Protected)* | **0.9929** | **0.7079** | **0.6915** | **0.6996** | **0.5380** | 95.2% | 70.0% | 1.50 | 1.40 |
| **leather** | 0.9236 | 0.5366 | 0.5249 | 0.5307 | 0.3612 | 72.8% | 100.0% | 2.20 | 2.71 |
| **zipper** | 0.8981 | 0.5152 | 0.6460 | 0.5732 | 0.4018 | 84.0% | 62.5% | 1.47 | 0.91 |
| **screw** | 0.8776 | 0.4175 | 0.5885 | 0.4884 | 0.3231 | 84.9% | 65.9% | 2.28 | 2.00 |
| **transistor** | 0.8200 | 0.4354 | 0.3898 | 0.4113 | 0.2589 | 62.5% | 76.7% | 2.69 | 0.88 |

### 4.2 Unbiased Benchmark with Locked Normal Calibration (Phase 3.1)
*Saved in:* `results/phase3/final_test/`
> **Methodology Note:** Thresholds were calibrated strictly on held-out normal images and locked BEFORE evaluation on the test set. Zero test-set ground truth peeking occurred.

| Category | Image AUROC | Pixel Precision | Pixel Recall | Pixel F1 | Pixel IoU | Detection Rate | Normal Accuracy | Locked $T_{image}$ | Locked $T_{pixel}$ | Spatial Prior |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** | **0.9929** | 0.4118 | **0.8951** | **0.5641** | **0.3928** | **96.8%** (61/63) | 95.0% (19/20) | 2.06 | 0.66 | Enabled |
| **leather** | **0.9239** | 0.1074 | 0.7872 | 0.1890 | 0.1044 | 72.8% (67/92) | 96.9% (31/32) | 2.96 | 1.71 | Disabled |
| **zipper** | **0.8981** | 0.2553 | 0.3619 | 0.2994 | 0.1760 | 33.6% (40/119) | **100.0%** (32/32) | 2.72 | 0.57 | Enabled |
| **screw** | **0.8776** | 0.1945 | 0.4065 | 0.2631 | 0.1515 | 42.9% (51/119) | **100.0%** (41/41) | 2.95 | 1.66 | Enabled |
| **transistor** | **0.8200** | **0.6386** | 0.1059 | 0.1817 | 0.0999 | 15.0% (6/40) | **100.0%** (60/60) | 3.41 | 1.18 | Enabled |


### 4.3 Protection of Bottle Phase 2.3 Baseline
The Phase 2.3 bottle benchmark (`AUROC=0.9929`, `Precision=0.7079`, `Recall=0.6915`, `F1=0.6996`, `IoU=0.5380`) remains completely protected in `models/bottle/patchcore_v23/` and historical project records. Phase 3.1 reports the locked calibration evaluation as an independent experiment without modifying baseline weights.

---

## 5. Web Application & API Workflow

### 5.1 FastAPI Backend Stack (`app/backend.py`)
- `GET /`: Service metadata, operational status, and `/docs` pointer.
- `GET /health`: Active compute device, loaded models in memory cache.
- `GET /categories`: Parameter specifications and model paths for the 5 categories.
- `POST /predict`:
  - Accepts multipart `file` and `category`.
  - Supports `PNG`, `JPG`, `JPEG`, `WEBP`, `BMP`.
  - Rejects empty uploads, corrupt binaries, and unsupported extensions (400 Bad Request).
  - Sanitizes response payload: no local machine usernames, OS paths, or environment paths are exposed.
- Swagger Documentation accessible at `http://127.0.0.1:8000/docs`.

### 5.2 Streamlit User Interface (`app/app.py`)
- **Primary Upload**: Drag-and-drop or browse files from local machine.
- **Initial State**: Starts with `uploaded_image = None`, showing *"Upload an image to begin inspection."*
- **State Management**:
  - Image hash and category tracking in `st.session_state`.
  - Changing category or uploading a new image immediately clears previous predictions, heatmaps, and bounding boxes.
- **Category Notice**:
  - Prominently warns the operator that the selected category must match the product in the image.
- **Result Presentation**:
  - **NORMAL**: Clear green status card, anomaly score vs. threshold, 0 localized defect regions, original image, no spurious red defect boxes.
  - **DEFECTIVE**: Red status card, anomaly score vs. threshold, number of defect regions, side-by-side original image vs. heatmap localization overlay with red bounding boxes, and region details table (coordinates, area, score, aspect ratio).
