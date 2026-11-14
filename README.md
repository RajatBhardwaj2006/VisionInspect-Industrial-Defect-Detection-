# VisionInspect — Industrial Defect Detection & Localization

[![Maintainer](https://img.shields.io/badge/Maintainer-RajatBhardwaj2006-blue.svg)](https://github.com/RajatBhardwaj2006)
[![GitHub Repository](https://img.shields.io/badge/GitHub-VisionInspect-green.svg)](https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git)
[![Live Demo](https://img.shields.io/badge/Streamlit%20Cloud-Live%20App-FF4B4B.svg)](https://visioninspect.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12%20(64--bit)-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-orange.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-121%2F121%20Passing-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-blue.svg)](https://git-lfs.github.com/)

> **Live Deployment:** [https://visioninspect.streamlit.app](https://visioninspect.streamlit.app)

VisionInspect is an unsupervised computer vision framework for industrial quality assurance and sub-pixel defect localization. Engineered for physical assembly lines and automated inspection gates, VisionInspect detects structural, surface, and cosmetic departures from nominal factory baselines **without requiring defective samples for training**.

---

## 1. Project Introduction

In modern manufacturing, defective components are rare, unpredictable, and varied. Traditional supervised deep learning models require thousands of curated defect images with ground-truth segmentation masks for every defect mode (scratches, cracks, cuts, missing parts, deformities). This is fundamentally impractical and cost-prohibitive in high-mix, low-volume production lines.

**VisionInspect solves this through unsupervised nominal representation modeling:**
- **Zero Defect Training:** Models are trained exclusively on nominal (defect-free) components to learn the natural manifold of standard parts.
- **Deep PatchCore v2.3 Engine:** Deep visual feature representations are extracted across multiple hierarchical layers of a pre-trained ResNet-18 backbone and searched against a minimax greedy coreset memory bank.
- **Explainable Anomaly Localization:** Out-of-distribution visual departures are scored continuously, refined with category-specific spatial background priors, and segmented into bounding boxes and surface area metrics.
- **Dual-Gated Quality Gate:** Anomaly decisions require both the continuous image score to exceed a statistically calibrated threshold ($T_{\text{image}}$) and connected anomaly regions to meet minimal geometric criteria ($A \ge \text{min\_area}$), eliminating single-pixel sensor noise.

---

## 2. Key Features

- **Unsupervised Anomaly Detection:** Learns nominal appearance from defect-free parts; flags unseen defect modes automatically.
- **Sub-Pixel Defect Localization:** High-resolution anomaly heatmaps and bounding boxes isolate defects on the physical component surface.
- **Dual Inference Architecture:**
  - **In-Process Engine (Streamlit Community Cloud):** Zero-overhead direct inference with lazy single-model memory purging for $0 cloud hosting within a 1 GB RAM footprint.
  - **REST API Backend (Local / Enterprise):** Decoupled high-throughput FastAPI service (`/predict`, `/predict/batch`, `/health`, `/categories`).
- **Explainable Quality Verdicts:** Three-tier plain-English explanations for shop-floor operators alongside full technical matrices for QA engineers.
- **Automated Inspection Reports:** Instant, production-grade PDF certification reports with embedded heatmap overlays, bounding-box geometry, and regulatory disclaimers.
- **Cross-Category Safety Guard:** Pre-inference centroid distance screening prevents cross-product false rejections (e.g., inspecting a screw under a bottle model).
- **Standalone Verification Suite:** 21 curated test samples and 121 automated regression tests run completely offline without downloading the 5 GB MVTec dataset.

---

## 3. Supported Production Categories

VisionInspect is calibrated across **five industrial manufacturing categories** from the MVTec Anomaly Detection benchmark:

| Category | Component Class | Physical Defect Modes Detected | Spatial Prior | Score Aggregation | Image Threshold ($T_{\text{image}}$) |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Bottle** | Rigid Glass Container | Mouth cracks, glass fractures, contamination | Yes ($p75$ baseline) | Max Raw Patch | `1.5000` |
| **Leather** | Flexible Natural Texture | Cuts, fold creases, dye flaws, glue, punctures | No (Uniform texture) | Max Raw Patch | `2.9000` |
| **Transistor** | Semiconductor Package | Bent leads, cut pins, damaged housing, missing body | No (Lead-frame) | Top-200 Average | `3.5250` |
| **Zipper** | Mechanical Fastener | Broken teeth, split tracks, fabric edge fraying | Yes (Weave prior) | Max Raw Patch | `1.4700` |
| **Screw** | Threaded Metal Hardware | Thread damage, head scratches, neck stripping | No (Rotational) | Top-100 Average | `2.8000` |

---

## 4. System Architecture

The end-to-end inspection pipeline flows from image acquisition to quality disposition:

```
[Input Image (224×224 RGB)]
            ↓
[Category Centroid Screening] ──(Mismatch Alert)──> [UI Warning Dialog]
            ↓ (Pass)
[ResNet-18 Multi-Scale Feature Extractor (Layers 1, 2, 3)]
            ↓
[448-Dimensional Local Patch Neighborhood Embeddings]
            ↓
[Greedy Coreset Memory Bank Search (10% Subsampling)]
            ↓
[Anomaly Heatmap Generation & Spatial Prior Normalization]
            ↓
[Connected-Component Morphology & Bounding Box Extraction]
            ↓
[Dual-Gated Inspection Verdict (Score + Region Area)]
            ↓
┌────────────────────────────────────────────────────────┐
│                   Inspection Outputs                   │
├───────────────────────────┬────────────────────────────┤
│ • Interactive Streamlit UI│ • QA Technical Details     │
│ • Plain-English Explanations│ • Downloadable PDF Report │
└───────────────────────────┴────────────────────────────┘
```

---

## 5. Technology Stack

- **Deep Learning Framework:** PyTorch `2.2.2+cpu`, Torchvision `0.17.2+cpu`
- **Feature Backbone:** ResNet-18 (ImageNet pre-trained feature pyramid pooling)
- **Computer Vision & Array Math:** OpenCV (`opencv-python 4.8+`), NumPy `1.26.4`, SciPy `1.12.0`, scikit-learn `1.4.2`
- **Web Application & UI:** Streamlit `1.32.0+` (custom dark/light high-contrast theme system)
- **Backend API:** FastAPI `0.110.0+`, Uvicorn `0.28.0+`, Pydantic
- **PDF Generation:** ReportLab `4.0.0+`
- **Configuration & Utilities:** PyYAML `6.0.3`, Pandas `2.2.0`, Pillow `10.2.0+`
- **Testing:** Pytest `8.0.0+` (121 automated tests)
- **Deployment Platform:** Streamlit Community Cloud (Linux, Python 3.12, system `packages.txt`)

---

## 6. Repository Structure

```text
VisionInspect/
├── .devcontainer/                     # VS Code Dev Container configuration
├── .dockerignore                      # Docker build exclusions
├── .env.example                       # Environment variable template
├── .gitattributes                     # Git LFS pointer rules for weights
├── .gitignore                         # Exclusions for datasets, cache, secrets
├── .python-version                    # Python 3.12 runtime pin for Streamlit Cloud
├── .streamlit/                        # Streamlit theme & server configuration
│   └── config.toml
├── Dockerfile                         # Production multi-stage Docker container build
├── LICENSE                            # MIT License
├── README.md                          # Comprehensive project documentation
├── config.yaml                        # Master thresholds, priors & morphology configs
├── inference.py                       # CLI inference entry point
├── packages.txt                       # Linux OS dependencies for OpenCV on cloud
├── render.yaml                        # Render cloud deployment blueprint
├── requirements.txt                   # Primary pinned dependencies
├── requirements-lock.txt              # Frozen Python 3.12 environment snapshot
├── run_manual_inference.ps1           # Portable PowerShell test runner
│
├── app/                               # Application Layer
│   ├── app.py                         # Streamlit frontend & in-process PatchCore engine
│   ├── backend.py                     # FastAPI REST server (/health, /predict)
│   ├── components/                    # UI modular components
│   │   ├── footer.py                  # Responsive footer
│   │   └── model_docs.py              # Technical documentation viewer
│   ├── utils/                         # Application utilities
│   │   └── pdf_report.py              # Automated ReportLab PDF generator
│   └── assets/icons/                  # Brand 3D logos and favicons
│
├── assets/                            # Static Visual Assets
│   ├── carton_perfect.png             # Landing page illustrative nominal sample
│   ├── carton_damaged.png             # Landing page illustrative defect sample
│   └── test_samples/                  # 21 curated offline test samples
│       ├── bottle/                    # (broken_large, broken_small, contamination, good)
│       ├── leather/                   # (cut, fold, color, good)
│       ├── screw/                     # (thread_side, scratch_head, manipulated, good)
│       ├── transistor/                # (bent_lead, cut_lead, damaged_case, good)
│       └── zipper/                    # (broken_teeth, rough, split_teeth, good)
│
├── Docs/                              # Technical Engineering Documentation
│   ├── REPOSITORY_CLEANUP_AUDIT.md    # Pre-submission cleanup audit and inventory
│   ├── SECURITY_AUDIT.md              # Production security audit (OWASP, paths, MIME)
│   ├── DEPLOYMENT.md                  # Cloud & local deployment guide
│   ├── DATASET_SETUP.md               # Guide to MVTec AD dataset setup
│   └── category_compatibility.md      # Category mismatch defense specification
│
├── models/                            # Production Model Checkpoints
│   ├── category_prototypes.pt         # 5-category centroid embeddings (11 KB)
│   ├── bottle/patchcore_v23/          # Memory bank, spatial prior, metadata
│   ├── leather/patchcore_v23/         # Memory bank, spatial prior, metadata
│   ├── transistor/patchcore_v23/      # Memory bank, spatial prior, metadata
│   ├── zipper/patchcore_v23/          # Memory bank, spatial prior, metadata
│   ├── screw/patchcore_v23/           # Memory bank, spatial prior, metadata
│   └── baseline/bottle/               # Phase 1 autoencoder baseline config
│
├── notebooks/                         # Exploratory Data Analysis & Research
│   ├── 01_dataset_exploration.ipynb   # Initial dataset distributions
│   ├── 03_autoencoder_training.ipynb  # Baseline autoencoder experimentation
│   └── 04_anomaly_detection.ipynb     # PatchCore density estimation research
│
├── results/                           # Verified Experimental & Evaluation Results
│   ├── evaluation/                    # Per-category evaluation JSONs
│   ├── phase1_final/                  # Phase 1 benchmark comparisons
│   ├── phase2/                        # Phase 2 crack localization comparisons
│   ├── phase3/                        # Phase 3 multi-category evaluation metrics
│   └── Updated_model_predictions/     # Locked per-sample audit report & summary
│
├── scripts/                           # Engineering Utilities & Automation
│   ├── verify_models.py               # Model integrity and Git LFS validator
│   ├── setup_windows.ps1              # Automated Windows environment installer
│   ├── start_visioninspect.ps1        # Automated backend + frontend launcher
│   ├── evaluate_model.py              # CLI batch evaluation script
│   └── train_patchcore.py             # PatchCore memory bank training script
│
├── src/                               # Core Computer Vision Framework
│   ├── data/                          # Dataset loaders & preprocessing transforms
│   ├── detection/                     # PatchCore v2.3 detector, scoring, morphology
│   ├── models/                        # Feature extractors (ResNet-18 pyramid pooling)
│   ├── evaluation/                    # AUROC, ROC curves, pixel sweeps, visualizations
│   └── utils/                         # Config loaders and path helpers
│
└── tests/                             # Automated Test Suite (121 Passed)
    ├── test_app.py                    # App entry point contract
    ├── test_category_mismatch.py      # Cross-category mismatch defense tests
    ├── test_dataset.py                # Dataset loaders and sample resolution
    ├── test_detection.py              # Anomaly detection contracts
    ├── test_footer.py                 # UI footer layout
    ├── test_frontend_state.py         # Frontend state atomic contracts
    ├── test_inspect_page.py           # Catalog integrity and golden samples
    ├── test_model.py                  # Autoencoder architecture
    ├── test_model_dispatch.py         # Version dispatching (v2.2 vs v2.3)
    ├── test_model_docs.py             # Technical documentation assertions
    ├── test_patchcore*.py             # PatchCore memory bank algorithms
    ├── test_pdf_report.py             # PDF generation and table formatting
    ├── test_phase3_*.py               # Phase 3 multi-category regression tests
    └── test_security_hardening.py     # Path traversal, upload limits, sanitization
```

---

## 7. Installation & Setup

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python:** **Python 3.12 (64-bit)** (Python 3.12.10 recommended)
- **Git & Git LFS:** Git 2.30+ with [Git LFS](https://git-lfs.github.com/) installed
- **Hardware:** Minimum 4 GB RAM (CPU inference supported; CUDA GPU optional)

### Step 1: Clone Repository & Pull Model Weights
```bash
git clone https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git
cd VisionInspect-Industrial-Defect-Detection-
git lfs pull
```

### Step 2: Create Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> *Note for CPU-only PyTorch:* If installing manually on a minimal machine:
> ```bash
> pip install torch==2.2.2+cpu torchvision==0.17.2+cpu --index-url https://download.pytorch.org/whl/cpu
> ```

### Step 4: Verify Model Checkpoints
```bash
python scripts/verify_models.py
```
This script verifies that all 5 production PatchCore models and category prototypes exist, are valid PyTorch tensors, and are not unpulled Git LFS pointers.

---

## 8. Local Execution

### Option A: Standalone In-Process Mode (Fastest & Simplest)
Launch VisionInspect directly using the built-in in-process PatchCore engine:
```bash
python -m streamlit run app/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### Option B: Decoupled REST Service (Production Simulation)
**Terminal 1 — Start FastAPI Backend:**
```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
```
API Documentation will be live at: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

**Terminal 2 — Start Streamlit Frontend:**
```powershell
python -m streamlit run app/app.py --server.port 8501
```

### Option C: Windows Automated Script
```powershell
.\scripts\setup_windows.ps1
.\scripts\start_visioninspect.ps1
```

---

## 9. Model Architecture & Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Type** | PatchCore v2.3 (Unsupervised Density-based Anomaly Localization) |
| **Feature Extractor** | ResNet-18 (ImageNet pre-trained weights) |
| **Pooled Layers** | Layer 1 (64D), Layer 2 (128D), Layer 3 (256D) |
| **Patch Feature Dimension** | 448-dimensional dense representations |
| **Spatial Grid** | $64 \times 64$ grid per image (4,096 patches) |
| **Memory Bank Subsampling** | 10% minimax greedy coreset subsampling |
| **Distance Metric** | Chunked Euclidean distance (memory-bounded matrix operations) |
| **Decision Rule** | Dual-Gated: $[S_{\text{image}} > T_{\text{image}}] \land [\text{Area} \ge \text{min\_area}]$ |

---

## 10. Verified Benchmark Evaluation

All models were evaluated on official held-out test splits from the MVTec AD benchmark, with calibration performed strictly on 20% validation nominal splits (zero test-set leakage):

| Category | Image-Level AUROC | Normal Specificity | Defect Sensitivity | False Positive Rate | Peak Memory Bank Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bottle** | **100.0%** | 100.0% | 100.0% | 0.0% | 85,606 patches |
| **Leather** | **100.0%** | 100.0% | 100.0% | 0.0% | 100,352 patches |
| **Transistor** | **99.1%** | 97.5% | 98.4% | 2.5% | 87,244 patches |
| **Zipper** | **99.8%** | 100.0% | 98.9% | 0.0% | 98,304 patches |
| **Screw** | **99.4%** | 98.2% | 97.8% | 1.8% | 131,072 patches |
| **Macro Average** | **99.66%** | **99.14%** | **99.02%** | **0.86%** | **100,516 patches** |

*Detailed benchmark sweeps, ROC plots, and per-defect breakdowns are preserved in [`results/`](results/).*

---

## 11. Deployment

### Live Web Application
- **URL:** [https://visioninspect.streamlit.app](https://visioninspect.streamlit.app)
- **Host:** Streamlit Community Cloud (Zero-Cost Hosting Target)
- **Engine Architecture:** Direct in-process PatchCore v2.3 inference engine.
- **Memory Optimization:** Single-model lazy loading with automatic garbage collection and `torch.cuda.empty_cache()` equivalent tensor purging when switching categories, maintaining runtime memory comfortably below 700 MB.
- **System Dependencies:** Managed via `packages.txt` (`libgl1`, `libglib2.0-0`) and `.python-version` (pinned to Python 3.12).

---

## 12. Automated Test Suite

VisionInspect includes a rigorous test suite of **121 automated tests** covering machine learning inference, API contracts, PDF generation, security hardening, and UI state integrity.

To execute the complete test suite:
```bash
python -m pytest tests/ -v
```

Expected result:
```text
================= 121 passed, 1 warning in ~115s =================
```

### Test Scope
- `test_inspect_page.py`: Golden sample integrity, category catalog verification (strictly 5 categories).
- `test_patchcore_v23.py`: PatchCore v2.3 inference correctness and spatial priors.
- `test_category_mismatch.py`: Centroid distance screening and cross-category alert dialogs.
- `test_security_hardening.py`: Path traversal protection, Pillow decompression-bomb defense, upload MIME validation.
- `test_pdf_report.py`: ReportLab automated table and PDF generation.
- `test_frontend_state.py`: Atomic session state and decision margin calculations.

---

## 13. System Limitations

- **Optical Anomaly Screening:** VisionInspect provides optical visual anomaly detection. It flags visual irregularities but does not replace physical structural, electrical, or tensile certification.
- **Lighting & Fixture Alignment:** Unsupervised patch distance relies on consistent imaging conditions. Significant ambient illumination fluctuations or extreme orientation shifts can elevate patch distances.
- **Model Scope:** Models are calibrated exclusively for the 5 documented industrial categories (Bottle, Leather, Transistor, Zipper, Screw). Inspecting arbitrary consumer objects outside this distribution will trigger category mismatch warnings.
- **Memory-Constrained Single-Model Concurrency:** On resource-constrained free cloud tiers (e.g., 1 GB RAM), models are loaded lazily one category at a time to prevent out-of-memory container termination.

---

## 14. License

This project is open-source software licensed under the **MIT License**. See the [LICENSE](LICENSE) file for complete terms and permissions.
