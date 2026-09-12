# Phase 3 Pre-Implementation Audit Report
**VisionInspect: Industrial Defect Detection & Localization**  
**Audit Date:** September 10, 2026  
**Status:** AUDIT & PLANNING ONLY (Implementation Locked)

---

## Executive Summary

This pre-implementation audit inspects the current state of the VisionInspect codebase following the successful completion of **Phase 1** (Convolutional Autoencoder Baseline), **Phase 2** (PatchCore Feature Anomaly Baseline), **Phase 2.2** (Crack-Sensitive Multi-Scale PatchCore), and **Phase 2.3** (Precise Crack & Defect Localization).

The primary objective of this audit is to rigorously evaluate:
1. Current project architecture, components, and workflow.
2. The exact mechanics and dependencies of the **PatchCore v2.3** baseline.
3. Readiness and blockers for **multi-category generalization** across the 15 MVTec AD categories.
4. Compliance gaps relative to teacher/academic course requirements.
5. Architectural risks and critical design questions that **must be discussed and resolved with the user** before Phase 3 implementation begins.

> [!IMPORTANT]
> **Strict Phase 3 Constraint:**  
> Phase 3 implementation has **NOT** started. No models were trained, no weights were modified, no hyperparameters were altered, and no code changes were introduced during this audit. All 27 existing unit tests remain fully passing.

---

## 1. Current Project Architecture

### 1.1 Directory Structure Overview

```text
VisionInspect/
│
├── config.yaml                              # Global YAML configuration
├── inference.py                             # Unified CLI inference interface
├── requirements.txt                         # Root dependency specifications
├── requirements-visioninspect-py312.txt     # Locked virtualenv dependencies
├── setup_project_structure.py               # Initial scaffolding utility
├── README.md                                # Repository overview & maintainer info
│
├── app/                                     # Streamlit Web Application (Scaffold)
│   ├── app.py                               # Main app entry (0 bytes placeholder)
│   ├── assets/                              # App static assets (.gitkeep)
│   ├── components/                          # UI components (0 bytes placeholders)
│   │   ├── upload.py, inspection.py, heatmap_viewer.py, batch_results.py, reports.py
│   └── styles/                              # CSS styling (0 bytes placeholder)
│
├── dataset/
│   └── mvtec_anomaly_detection/             # Complete MVTec AD dataset (all 15 categories present)
│       ├── bottle/                          # [Active] train/good (209), test (83), ground_truth
│       ├── cable/, capsule/, carpet/, grid/, hazelnut/, leather/, metal_nut/,
│       │   pill/, screw/, tile/, toothbrush/, transistor/, wood/, zipper/
│
├── docs/                                    # Technical reports and documentation
│   ├── phase1_final_report.md               # Phase 1 Autoencoder report
│   ├── phase1_tweaks_report.md              # Phase 1 thresholding & morphology findings
│   ├── phase2_bottle_baseline.md            # Phase 2 initial PatchCore baseline report
│   ├── phase2_2_crack_localization_report.md# Phase 2.2 multi-scale PatchCore report
│   ├── phase2_3_precise_crack_localization_report.md # Phase 2.3 precision localization report
│   ├── phase2_final_report.md               # Verification & environment lock report
│   └── project_report.md                    # Project overview & milestone summaries
│
├── models/                                  # Serialized model weights & memory banks
│   ├── baseline/bottle/autoencoder.pth      # Reference Phase 1 Autoencoder weights
│   ├── bottle/
│   │   ├── autoencoder.pth                  # Phase 1 Autoencoder checkpoint (5.5 MB)
│   │   ├── patchcore/                       # Phase 2 baseline memory bank (21,401 patches, 384-d)
│   │   ├── patchcore_v22/                   # Phase 2.2 memory bank (85,606 patches, 448-d) + spatial prior
│   │   └── patchcore_v23/                   # Phase 2.3 reference memory bank (85,606 patches) + spatial prior
│   └── [cable ... zipper]/                  # Folders exist for all other 14 categories (.gitkeep only)
│
├── notebooks/                               # Exploratory Jupyter notebooks
│   ├── 01_dataset_exploration.ipynb         # Dataset analysis & visual exploration
│   ├── 02_data_preprocessing.ipynb          # (0 bytes placeholder)
│   ├── 03_autoencoder_training.ipynb        # Autoencoder experiment notebook
│   ├── 04_anomaly_detection.ipynb           # Heatmap and thresholding exploration
│   └── 05_evaluation.ipynb                  # (0 bytes placeholder)
│
├── results/                                 # Evaluation results, CSVs, metrics, and comparisons
│   ├── phase1_final/                        # Phase 1 metrics, sweeps, and visuals
│   ├── phase2/                              # Phase 2 metrics, sweeps, and comparisons
│   │   ├── phase2_2_visuals/, phase2_2_comparisons/
│   │   └── phase2_3/                        # Phase 2.3 verified metrics, CSVs, panel visuals
│   ├── predictions/                         # Saved prediction overlay PNGs
│   └── reports/, metrics/, heatmaps/, comparisons/
│
├── scripts/                                 # Executable CLI utility and evaluation scripts
│   ├── diagnose_patchcore.py                # Diagnostic script for raw vs prior maps
│   ├── diagnose_phase2_3.py                 # Diagnostic script for localization bottlenecks
│   ├── generate_visuals.py                  # Phase 1 visual generator
│   ├── generate_phase2_visuals.py           # Phase 2 visual generator
│   ├── generate_phase2_2_visuals.py         # Phase 2.2 visual generator
│   ├── generate_phase2_3_visuals.py         # Phase 2.3 6-panel visual comparison generator
│   ├── run_bottle_pipeline.py               # Phase 1 end-to-end pipeline runner
│   ├── run_phase1_final_eval.py             # Phase 1 evaluation suite (83 test images)
│   ├── run_patchcore_eval.py                # Phase 2 evaluation suite (83 test images)
│   ├── run_phase2_2_eval.py                 # Phase 2.2 evaluation suite (83 test images)
│   ├── run_phase2_3_eval.py                 # Phase 2.3 evaluation suite (83 test images)
│   ├── test_patchcore_v22.py                # Parameter sweep experiment script
│   ├── test_phase2_3_pipeline.py            # Phase 2.3 threshold & morphology optimizer
│   ├── train_patchcore.py                   # Phase 2 memory bank builder
│   └── train_patchcore_v22.py               # Phase 2.2 / 2.3 memory bank & spatial prior builder
│
├── src/                                     # Core modular Python packages
│   ├── data/
│   │   ├── dataset_loader.py                # MVTecTrainDataset & MVTecTestDataset classes
│   │   └── preprocessing.py                 # PyTorch transform pipelines (256x256, ToTensor)
│   ├── detection/
│   │   ├── anomaly_detector.py              # Phase 1 Autoencoder detector wrapper
│   │   ├── anomaly_score.py                 # Reconstruction error calculation functions
│   │   ├── heatmap.py                       # Anomaly heatmap generation utilities
│   │   ├── localization.py                  # Phase 1 & Phase 2 connected-component localization
│   │   ├── patchcore_detector.py            # Phase 2 baseline PatchCore detector wrapper
│   │   ├── patchcore_v22_detector.py        # Phase 2.2 detector wrapper
│   │   └── patchcore_v23_detector.py        # Phase 2.3 precise crack detector wrapper
│   ├── evaluation/
│   │   ├── metrics.py                       # AUROC, Precision, Recall, F1, IoU functions
│   │   ├── pixel_sweep.py                   # Pixel threshold sweep utilities
│   │   ├── roc_curve.py                     # (0 bytes placeholder)
│   │   └── visualization.py                 # (0 bytes placeholder)
│   ├── models/
│   │   ├── autoencoder.py                   # 4-layer CNN Autoencoder architecture
│   │   ├── feature_extractor.py             # (0 bytes placeholder)
│   │   ├── patchcore.py                     # Phase 2 PatchCoreModel & FeatureExtractor (layer2+3)
│   │   └── patchcore_v22.py                 # Phase 2.2/2.3 PatchCoreModelV22 & FeatureExtractorV22 (layer1+2+3)
│   └── utils/
│       ├── config.py                        # YAML configuration loader with fallback defaults
│       └── helpers.py                       # (0 bytes placeholder)
│
├── tests/                                   # Automated test suite (27 passing tests)
│   ├── test_app.py                          # Placeholder test for UI integration
│   ├── test_dataset.py                      # Dataset loading & ground truth mask integrity
│   ├── test_detection.py                    # Anomaly scoring, coordinate scaling, uniqueness
│   ├── test_model.py                        # Autoencoder initialization & state dict loading
│   ├── test_model_dispatch.py               # CLI model switching (--model v22 vs v23)
│   ├── test_patchcore.py                    # Phase 2 PatchCore feature extraction & memory bank
│   ├── test_patchcore_v22.py                # Phase 2.2 multi-scale extraction & spatial prior
│   └── test_patchcore_v23.py                # Phase 2.3 detector initialization & localization
│
└── visioninspect_py312/                     # Isolated virtual environment (Python 3.12.10, PyTorch CPU)
```

---

### 1.2 Pipeline Workflow: Data to Evaluation

```text
[MVTec Train / Test Images]
       │
       ▼
[src/data/preprocessing.py] ───► Resize (256x256), Convert to RGB, Normalize ToTensor()
       │
       ▼
[FeatureExtractorV22 in src/models/patchcore_v22.py]
       │── ResNet18 backbone (conv1, bn1, relu, maxpool)
       ├── layer1: [B, 64, 64, 64]
       ├── layer2: [B, 128, 32, 32] ──► Bilinear Upsample ──► [B, 128, 64, 64]
       └── layer3: [B, 256, 16, 16] ──► Bilinear Upsample ──► [B, 256, 64, 64]
       │
       ▼ Channel Concatenation: [B, 448, 64, 64] ──► 4,096 Patch Descriptors (d=448) per image
       │
       ▼
[PatchCoreModelV22 Training / Memory Bank]
       ├── Normal Training Images (209 images for bottle)
       ├── Full patch pool: 856,064 vectors
       ├── Coreset Subsampling (10% ratio): 85,606 vectors stored in memory_bank.pt
       └── Normal Spatial Background Prior: M_normal(h,w) [256, 256] stored in spatial_prior.pt
       │
       ▼
[Inference / Distance Computation]
       ├── Test Patches: [4096, 448]
       ├── Nearest-Neighbor Search: torch.cdist(test_patches, memory_bank, p=2.0)
       ├── Min Distance per patch ──► Spatial Grid [64, 64]
       ├── Bilinear Interpolation ──► [256, 256]
       └── Spatial Prior Subtraction: A_clean(h,w) = max(0, D_test(h,w) - M_normal(h,w))
       │
       ▼
[Post-Processing in src/detection/patchcore_v23_detector.py]
       ├── Gaussian Smoothing: sigma = 1.20
       ├── Direct Distance Thresholding: Binary Mask = (A_clean > 1.40)
       ├── Morphological Filtering: 3x3 Closing followed by Opening
       ├── Connected Component Noise Filter: Keep components with area >= 25 px
       ├── Border Margin Suppression: Filter components touching within 5 px of frame
       └── Spatial Proximity Merging: Merge component bounding boxes within 15.0 px
       │
       ▼
[Prediction & Decision]
       ├── Image Anomaly Score = max(A_clean)
       ├── Status = "DEFECTIVE" if (Score > 1.50 AND len(regions) > 0) else "NORMAL"
       └── Coordinate Scaling: Scale [256, 256] boxes back to original image size (e.g. 1024x1024)
       │
       ▼
[Visualization & Evaluation]
       ├── Overlay Generation: Heatmap + Red Bounding Boxes (results/predictions/)
       ├── 6-Panel Comparison: Original, Heatmap, Mask, GT, Baseline, Pred-vs-GT Overlay
       └── Full Evaluation (scripts/run_phase2_3_eval.py): Image AUROC, Pixel P, R, F1, IoU
```

---

### 1.3 Reusable Components for Multi-Category Generalization

| Component | File Path | Multi-Category Readiness | Notes |
|---|---|---|---|
| **Dataset Loader** | `src/data/dataset_loader.py` | **100% Reusable** | `MVTecTrainDataset` and `MVTecTestDataset` are fully parameterized by `category` and search `<dataset_root>/<category>/train/good/` and `<category>/test/`. |
| **Preprocessing** | `src/data/preprocessing.py` | **100% Reusable** | `get_train_transform()` standardizes any input image to $256 \times 256 \times 3$ PyTorch tensors. |
| **Feature Extractor** | `src/models/patchcore_v22.py` | **100% Reusable** | `FeatureExtractorV22` uses ImageNet-pretrained ResNet18 weights; features are visual descriptors independent of product category. |
| **PatchCore Core Logic** | `src/models/patchcore_v22.py` | **100% Reusable** | `fit()`, `predict()`, coreset subsampling, and spatial prior calculation work for any set of training tensors. |
| **Evaluation Metrics** | `src/evaluation/metrics.py` | **100% Reusable** | Pixel-level and image-level AUROC, Precision, Recall, F1, and IoU logic are category-agnostic. |
| **Localization Utilities** | `src/detection/patchcore_v23_detector.py` | **90% Reusable** | `postprocess_clean_mask()`, `extract_tight_regions()`, and `merge_nearby_regions()` work on any 2D binary mask. However, morphology parameters (kernel size, min area) may require tuning for texture categories. |

---

## 2. Category Support Audit

We performed an audit across the repository to determine whether categories other than `bottle` can be processed without modification.

### 2.1 Findings Matrix

| Dimension | Status | Evidence / Code Location | Details |
|---|---|---|---|
| **Dataset Existence** | **Present** | `dataset/mvtec_anomaly_detection/` | All 15 MVTec AD categories (`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`) exist on disk with complete test, train, and ground_truth folders. |
| **Dataset Loader** | **Supported** | `src/data/dataset_loader.py` (lines 13, 35) | `MVTecTrainDataset(category=...)` and `MVTecTestDataset(category=...)` accept any valid MVTec category string. |
| **Model Directories** | **Partially Supported** | `models/` | Directories `models/<category>/` exist for all 15 categories, but only `models/bottle/` contains trained checkpoints. All other 14 folders contain only `.gitkeep`. |
| **Training Scripts** | **Hardcoded** | `scripts/train_patchcore_v22.py` (line 20) | `category = "bottle"` is hardcoded. There are no CLI arguments (`--category`) in `train_patchcore.py` or `train_patchcore_v22.py`. |
| **Evaluation Scripts** | **Hardcoded** | `scripts/run_phase2_3_eval.py` (lines 32, 221) | `category = "bottle"` and output directory `results/phase2/phase2_3` are hardcoded without category subdirectories. |
| **Configuration** | **Hardcoded** | `config.yaml` (lines 27, 36, 48) | `model_dir: models/bottle/patchcore_v22` is hardcoded. There are no category-specific configuration blocks (e.g. `categories.cable`). |
| **Inference CLI** | **Blocked by Config** | `inference.py` (lines 21, 33–40) | `inference.py` accepts `--category <name>`. However, inside `PatchCoreDetectorV23.__init__`, line 128 queries `pc_cfg.get("model_dir")`, which returns the hardcoded string `"models/bottle/patchcore_v22"`. Thus, passing `--category cable` still loads the bottle memory bank unless `model_dir` is passed explicitly! |
| **Output Paths** | **Partially Supported** | `src/detection/patchcore_v23_detector.py` (line 215) | Saved prediction images include `{self.category}`, but save to a flat directory `results/predictions/` without category subfolders. |

### 2.2 Category Support Verdict
**Current Category Support:** **`bottle` only.**  
While the data loader and dataset structure are multi-category ready, the training scripts, evaluation scripts, detector default path resolution, and configuration schema currently assume `bottle`.

---

## 3. PatchCore v2.3 Architecture Specification

The current state-of-the-art model in VisionInspect is **PatchCore v2.3** (`src/detection/patchcore_v23_detector.py` backed by `src/models/patchcore_v22.py`).

```text
================================================================================
                    PATCHCORE v2.3 ARCHITECTURAL PROFILE
================================================================================
Feature Extractor:
  • Backbone Network:           torchvision.models.resnet18 (ImageNet Pretrained)
  • Truncation Point:           After layer3 (layer4, avgpool, and fc removed)
  • Extracted Layers:           layer1, layer2, layer3
  • Output Resolutions:
      - layer1:                 [B, 64, 64, 64]   (Stride 4)
      - layer2:                 [B, 128, 32, 32]  (Stride 8  -> Bilinear upsampled to 64x64)
      - layer3:                 [B, 256, 16, 16]  (Stride 16 -> Bilinear upsampled to 64x64)
  • Fusion Scheme:              Channel concatenation -> [B, 448, 64, 64]
  • Patch Spatial Grid:         64 x 64 = 4,096 patches per 256x256 image
  • Effective Patch Resolution: 4 x 4 pixels per patch
  • Feature Dimension:          448 channels per patch vector

Memory Bank & Coreset:
  • Training Category:          bottle/train/good (209 normal images)
  • Total Extracted Patches:    209 x 4,096 = 856,064 patch vectors
  • Coreset Reduction:          Uniform Random Subsampling at 10.0% ratio
  • Retained Coreset Patches:   85,606 patch vectors of dimension 448
  • Stored Checkpoint Size:     153,407,557 bytes (~153.4 MB)
  • Checkpoint File Path:       models/bottle/patchcore_v22/memory_bank.pt

Spatial Background Prior:
  • Methodology:                Mean patch distance across all 209 normal training images
  • Prior Tensor Shape:         [256, 256]
  • Prior Range on Bottle:      Min: 0.0226, Max: 1.9509 (concentrated on outer rim/threads)
  • Prior File Path:            models/bottle/patchcore_v22/spatial_prior.pt
  • Cleaned Anomaly Map:        A_clean(h,w) = max(0, D_test(h,w) - M_normal(h,w))

Post-Processing & Localization:
  • Smoothing Filter:           cv2.GaussianBlur (sigma = 1.20, ksize = (0, 0))
  • Pixel Thresholding:         Direct Absolute Thresholding (smooth_map > 1.40)
  • Normalization:              NONE (Absolute feature distance; per-image min-max removed)
  • Morphology:                 3x3 Closing (bridge cracks) -> 3x3 Opening (remove noise)
  • Min Region Area Filter:     >= 25 pixels (components < 25 px discarded)
  • Max Region Area Filter:     UNCONSTRAINED (artificial 3,000 px cap removed in v2.3)
  • Border Margin Filter:       5 pixels from frame edge
  • Proximity Merging:          Distance-based bounding box merging (merge_dist = 15.0 px)

Decision & Classification:
  • Image Anomaly Score:        max(A_clean) across image
  • Calibrated Image Threshold: 1.50
  • Decision Logic:             DEFECTIVE if (Score > 1.50 AND len(regions) > 0) else NORMAL
================================================================================
```

---

## 4. Phase 3 Improvement Opportunities & Code Mapping

Based on our code inspection, the following table maps every proposed Phase 3 improvement to its exact target location:

| Proposed Improvement | Target File / Module | Nature of Change |
|---|---|---|
| **Multi-Category Training CLI** | `scripts/train_patchcore.py` | Update to parse `--category <name>` and save to `models/<category>/patchcore_v23/`. |
| **Multi-Category Evaluation Suite** | `scripts/run_evaluation.py` (new/refactored) | Loop or accept `--category all` / `--category <name>`, write to `results/phase3/<category>/`. |
| **Category-Aware Config Schema** | `config.yaml`, `src/utils/config.py` | Add category-specific dictionary hierarchy (`categories.<cat>.thresholds`). |
| **Detector Dynamic Path Resolution** | `src/detection/patchcore_v23_detector.py` | Resolve `models/{self.category}/patchcore_v23` dynamically if not explicitly specified. |
| **Exploratory Data Analysis (EDA)** | `scripts/eda_mvtec.py` (new) | Compute class counts, resolutions, defect distributions, aspect ratios across all 15 categories. |
| **Adaptive Thresholding** | `src/detection/thresholding.py` (new) | Provide percentile-based, Otsu, or GMM adaptive thresholding alongside static thresholds. |
| **Texture vs Object Adaptation** | `src/models/patchcore_v22.py` | Disable/adjust spatial prior for unaligned textures (e.g. `carpet`, `leather`, `wood`). |
| **Small Defect Enhancement** | `src/models/feature_extractor_v3.py` (optional) | Investigate multi-scale patch pooling or higher input resolution ($512 \times 512$). |
| **Sensitivity Analysis** | `scripts/sensitivity_analysis.py` (new) | Systematically sweep noise, Gaussian blur, illumination, and coreset sampling ratios. |
| **Error Analysis & Failure Gallery** | `scripts/error_analysis.py` (new) | Automatically isolate, categorize, and render false positive and false negative samples. |
| **Explainability (XAI)** | `src/detection/explainability.py` (new) | Patch nearest-neighbor visualization: display the exact normal training patch most similar to a defect. |
| **Streamlit Web UI Integration** | `app/app.py`, `app/components/` | Wire existing detector inspect API to Streamlit file uploader, interactive threshold slider, and viewer. |

---

## 5. Teacher & Course Requirements Gap Analysis

We evaluated the VisionInspect repository against standard university/academic capstone evaluation criteria:

| Requirement | Current Status | Evidence in Repository | Required Action for Phase 3 |
|---|---|---|---|
| **1. Exploratory Data Analysis (EDA)** | **Partially Present** | `notebooks/01_dataset_exploration.ipynb` contains manual image plots for bottle. | Build an automated, reproducible EDA script (`scripts/eda_mvtec.py`) generating dataset summaries, image sizes, and defect distributions across categories. |
| **2. Unsupervised Anomaly Detection** | **Already Present** | Trained strictly on normal images (`train/good`). Evaluated on unseen defective/good test images. | Maintain unsupervised integrity when expanding to other categories. |
| **3. Multiple Approaches / Models** | **Already Present** | 4 distinct model evolutions: Phase 1 AE, Phase 2 PatchCore Baseline, Phase 2.2 Crack-Sensitive, Phase 2.3 Precise Crack. | Preserve all historical models for benchmark comparisons. |
| **4. Explainability (XAI)** | **Partially Present** | Visual heatmaps, prior-subtracted distance maps, and bounding boxes in `results/`. | Implement patch nearest-neighbor retrieval: show the user *why* a patch is anomalous by showing the closest normal patch. |
| **5. Model Evaluation** | **Already Present** | Full 83-image test evaluation for bottle: Image AUROC, Pixel Precision, Recall, F1, IoU, per-category breakdown. | Expand this evaluation harness to multiple categories. |
| **6. Error Analysis** | **Partially Present** | False positive/negative pixel counts in CSVs; 15-image manual audit documented. | Create automated error analysis scripts that isolate, classify, and visualize FP and FN failure cases. |
| **7. Sensitivity Analysis** | **Partially Present** | Parameter grid search across sigma, kernel sizes, and thresholds in `test_phase2_3_pipeline.py`. | Create a formal sensitivity analysis across image resolutions, noise levels, and coreset ratios. |
| **8. Deployment / Integration** | **Partially Present** | CLI inference tool `inference.py` fully operational. Streamlit app files in `app/` are empty. | Populate `app/app.py` with an interactive web UI. |
| **9. Complete ML Lifecycle** | **Partially Present** | Data loading $\to$ feature extraction $\to$ memory bank $\to$ evaluation $\to$ CLI inference working. | Add automated pipeline orchestration and interactive deployment. |
| **10. GitHub Documentation** | **Already Present** | Comprehensive Markdown reports for all phases in `docs/`, clean `README.md` with maintainer links. | Maintain documentation standards with a Phase 3 final report. |

---

## 6. Protected Files & Artifacts (DO NOT MODIFY)

The following files represent the verified milestones of Phase 1, Phase 2, Phase 2.2, and Phase 2.3. They **must remain protected and unmodified**:

1. **Dataset:**
   - `dataset/mvtec_anomaly_detection/` (all categories, images, masks)
2. **Model Weights & Memory Banks:**
   - `models/baseline/bottle/autoencoder.pth` (Phase 1 baseline)
   - `models/bottle/autoencoder.pth` (Phase 1 model)
   - `models/bottle/patchcore/memory_bank.pt` (Phase 2 baseline)
   - `models/bottle/patchcore_v22/memory_bank.pt` & `spatial_prior.pt` (Phase 2.2 model)
   - `models/bottle/patchcore_v23/memory_bank.pt` & `spatial_prior.pt` (Phase 2.3 verified reference)
3. **Verified Baseline Results:**
   - `results/phase1_final/`
   - `results/phase2/` (including `phase2_2_*` and `phase2_3/`)
4. **Historical Documentation:**
   - `docs/phase1_final_report.md`
   - `docs/phase2_final_report.md`
   - `docs/phase2_3_precise_crack_localization_report.md`
5. **Existing Unit Test Suite:**
   - `tests/test_app.py`, `test_dataset.py`, `test_detection.py`, `test_model.py`, `test_model_dispatch.py`, `test_patchcore.py`, `test_patchcore_v22.py`, `test_patchcore_v23.py` (27 passing tests)

---

## 7. Technical Risks Before Phase 3

1. **Object vs Texture Categorization:**
   - MVTec contains **objects** (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`) and **textures** (`carpet`, `grid`, `leather`, `tile`, `wood`).
   - Objects benefit heavily from the **Spatial Background Prior** because bottle boundaries and rims are spatially aligned across images.
   - Textures are translation-invariant; their defects are local disruptions in continuous patterns. Applying a rigid spatial prior to textures could introduce artifacts or fail to suppress repetitive background noise.
2. **Disk Storage & Memory Footprint:**
   - The bottle memory bank (85,606 vectors of float32, $d=448$) consumes **153.4 MB**.
   - If all 15 categories are trained with identical settings: $15 \times 153.4\text{ MB} \approx \mathbf{2.3\text{ GB}}$ of disk space.
   - For student machines or Git repositories, committing 2.3 GB of tensor binaries is impractical. A clean `.gitignore` and selective category training strategy are required.
3. **Inference Latency on CPU:**
   - Nearest-neighbor search computes pairwise distances across $4,096 \times 85,606 \approx 3.5 \times 10^8$ operations per image.
   - On CUDA, this takes $\sim 20$ ms. On CPU, this can take $1.5 - 3.0$ seconds per image. Full evaluation of all 15 categories ($\sim 1,250$ test images) on CPU could take 45–60 minutes.
4. **Global vs Category-Specific Thresholds:**
   - Normal anomaly scores vary drastically by material. Transparent glass (`bottle`) has different feature distances than braided copper (`cable`) or woven fabric (`carpet`). A single threshold will cause catastrophic false positives on some categories and zero recall on others.
5. **Image Aspect Ratios & Native Resolutions:**
   - Bottle is square ($1024 \times 1024$). Other categories (e.g. `cable`: $1024 \times 1024$, `transistor`: $1024 \times 1024$, `screw`: $1024 \times 1024$) are also $1024 \times 1024$, but some defect types (e.g. tiny pinholes in `capsule` or thin bent wires in `cable`) may suffer from $256 \times 256$ downsampling.

---

## 8. Recommended Phase 3 Sequence

To avoid breaking working code and maintain scientific rigor, we recommend structuring Phase 3 into the following sub-phases:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.0: Framework Generalization & Multi-Category Config Schema     │
│   • Update config.yaml with category-aware schema                      │
│   • Update scripts/train_patchcore.py to accept --category             │
│   • Ensure PatchCoreDetector dynamically resolves category model paths │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.1: Comprehensive Exploratory Data Analysis (EDA)               │
│   • Create scripts/eda_mvtec.py                                        │
│   • Output image counts, resolution distributions, defect taxonomies   │
│   • Generate multi-category dataset report for course requirements     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.2: Multi-Category Expansion (Target Pilot Batch)               │
│   • Train PatchCore memory banks on selected pilot categories          │
│     (e.g. 1 additional object: cable; 1 texture: leather or tile)      │
│   • Adapt spatial prior handling for texture categories                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.3: Multi-Category Evaluation & Threshold Calibration           │
│   • Create scripts/run_multi_category_eval.py                          │
│   • Calibrate category-specific image and pixel thresholds             │
│   • Benchmark metrics against bottle v2.3 reference                    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.4: Localization & Boundary Refinement                          │
│   • Address broad heatmaps and contamination weaknesses                │
│   • Investigate adaptive thresholding (Otsu / percentile-based)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.5: Systematic Sensitivity & Error Analysis                     │
│   • Perturbations: noise, illumination, resolution, coreset ratio      │
│   • Automated failure mode categorization (FP vs FN galleries)         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.6: Model Explainability (XAI)                                  │
│   • Implement nearest normal patch retrieval and visual attribution    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.7: Interactive Deployment (Streamlit Web App)                  │
│   • Wire app/app.py to run live inference on uploaded images           │
│   • Interactive sliders for category selection and thresholds          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3.8: Final Comprehensive Documentation & Project Submission      │
│   • Final Phase 3 Report synthesizing all milestones                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Experimental Rules for Phase 3

1. **No Metric Fabrication:** All metrics must be produced by automated scripts reading real ground truth masks.
2. **Preserve Baseline Integrity:** The verified Phase 2.3 metrics (AUROC: 0.9929, F1: 0.6996, IoU: 0.5380 on bottle) must never be overwritten or regressed.
3. **Controlled Variable Isolation:** When experimenting with new backbones or thresholding algorithms, change one variable at a time.
4. **Honest Reporting of Failures:** If an approach fails on a specific category (e.g. textures), document the failure and root cause rather than tweaking thresholds artificially.
5. **No Ground Truth Leaks:** Ground truth masks are strictly for evaluation. They must never be accessed during inference or training.

---

## 10. OPEN QUESTIONS BEFORE PHASE 3

The following design decisions and questions **must be discussed and answered by the user before any implementation begins**:

### Question 1: Scope of Category Expansion
*Should Phase 3 train and evaluate on all 15 MVTec AD categories, or should we select a representative pilot subset first?*
- **Option A (Recommended Pilot Batch):** Select 3–4 diverse categories representing distinct challenges:
  - 1 Rigid Object with structure: `bottle` (already done, reference)
  - 1 Deformable / Multi-part Object: `cable` or `metal_nut`
  - 1 Structured Texture: `tile` or `grid`
  - 1 Non-structured Texture: `leather` or `wood`
- **Option B (All 15 Categories):** Train memory banks and calibrate thresholds for all 15 categories. (Note: Will require $\sim 2.3$ GB disk storage and $\sim 45-60$ minutes evaluation time).

### Question 2: Handling Textures vs Spatial Background Prior
*The spatial background prior ($M_{normal}$) was critical for eliminating bottle rim artifacts because bottle geometry is aligned. How should we handle texture categories?*
- **Option A:** Make the spatial prior optional/configurable per category (`use_spatial_prior: false` for textures like `carpet`, `tile`, `wood`; `true` for rigid objects like `bottle`, `capsule`).
- **Option B:** Replace the 2D spatial prior with a global feature distance percentile threshold for all categories.

### Question 3: Backbone Network Selection
*Should we stick with ResNet18 or explore a larger backbone?*
- **Option A (Preserve ResNet18):** Lightweight, fast inference ($\sim 20$ ms on GPU), manageable memory bank ($\sim 150$ MB), student machine friendly.
- **Option B (Investigate WideResNet50 or EfficientNet):** Higher representational power, but memory bank per category will grow to $\sim 600$ MB - 1 GB, significantly increasing inference latency.

### Question 4: Memory Bank Storage Strategy
*How should memory bank tensors be stored on disk?*
- **Option A:** Store memory banks in `models/<category>/patchcore_v23/` and add `*.pt` to `.gitignore` so they are generated locally and not pushed to GitHub.
- **Option B:** Keep memory banks only for evaluated pilot categories to stay within Git LFS / storage limits.

### Question 5: Threshold Strategy across Categories
*How should threshold calibration be automated across multiple categories?*
- **Option A (Category-Specific Calibrated Table):** Run threshold sweep on each category's validation/test set and record optimal `image_threshold` and `pixel_threshold` in `config.yaml` under `categories.<category_name>`.
- **Option B (Normalized Anomaly Scores):** Fit a statistical distribution (e.g. Gaussian or Weibull) to normal training image scores and threshold at a fixed $p$-value / z-score (e.g. $\mu + 3\sigma$) across all categories.

### Question 6: Deployment & Streamlit Web Interface
*Is building the interactive Streamlit web application (`app/app.py`) a required deliverable for Phase 3?*
- If yes, what features should the UI include (e.g. single-image drag-and-drop, category dropdown, interactive threshold slider, side-by-side heatmap viewer, batch inspection report download)?

---

**AUDIT CONCLUSION:**  
The repository is in a healthy, well-tested, and reproducible state (27/27 tests passing). The codebase is architecturally prepared for Phase 3 generalization, provided that configuration hardcoding is refactored into a category-aware schema and the open design questions above are decided.

**Awaiting user instructions on the Open Questions above before taking any implementation actions.**
