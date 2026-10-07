# VisionInspect — Team Handoff Architecture & Repository Audit

> **Document Type:** Production Architecture, Runtime Dependency & Repository Audit  
> **Target Audience:** Engineering Teammates, QA, Evaluators, Reviewers  
> **Environment Baseline:** Windows 10/11, Python 3.12 (64-bit), Git LFS  
> **Version:** 3.2.0 Locked

---

## 1. Project Architecture Overview

VisionInspect is an unsupervised optical defect detection and sub-pixel localization system engineered for industrial manufacturing. The application is architected around a dual-tier client-server topology:

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Web UI                     │
│                (app/app.py on port 8501)               │
└──────────────────────────┬─────────────────────────────┘
                           │ Multipart HTTP / REST
                           ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│             (app/backend.py on port 8000)              │
├──────────────────────────┬─────────────────────────────┤
│  Category Compatibility  │      PatchCore v2.3         │
│  Checker (ResNet18 448D) │   Detector Dispatch Engine  │
└──────────────────────────┴──────────────┬──────────────┘
                                          │ Memory Bank Matching
                                          ▼
                      ┌───────────────────────────────────────┐
                      │          models/<category>/           │
                      │ ├── memory_bank.pt (Coreset vectors)  │
                      │ ├── spatial_prior.pt (Normal context) │
                      │ └── metadata.json (Thresholds/params) │
                      └───────────────────────────────────────┘
```

1. **Frontend (`app/app.py`)**:
   - Streamlit single-page application with modular routing (`Home`, `Inspect`, `Models`, `About`).
   - Implements Apple-industrial minimalist design tokens with automatic Day/Night theme synchronization.
   - Provides drag-and-drop file upload, quick sample selection from `assets/test_samples/`, multi-image batch queue, interactive heatmaps, tight bounding box localization, usability assessments, and automated PDF inspection reports via ReportLab.

2. **Backend REST API (`app/backend.py`)**:
   - High-throughput asynchronous FastAPI server on port 8000.
   - Routes:
     - `GET /` — Service status & endpoint catalog.
     - `GET /health` — Health check reporting status, loaded models, and compute device.
     - `GET /categories` — Supported category schema and active configurations.
     - `POST /inspect` — Core inference endpoint (accepts image + category, returns scores, heatmaps, bounding boxes, and usability disposition).
     - `POST /check_category` — Centroid-based prototype distance check (<50ms) to detect mismatched components before running full defect inference.

---

## 2. Service Startup Specifications

### Backend (FastAPI)
- **Entry point:** `app/backend.py`
- **ASGI Server:** Uvicorn
- **Command:**
  ```powershell
  python -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
  ```
- **Compute device:** Automatic selection (`cuda` if NVIDIA GPU with CUDA is present, otherwise `cpu`).
- **Health check:** `http://127.0.0.1:8000/health` (responds with `{"status": "ok", ...}`)

### Frontend (Streamlit)
- **Entry point:** `app/app.py`
- **Server:** Streamlit
- **Command:**
  ```powershell
  python -m streamlit run app/app.py --server.port 8501
  ```
- **Access URL:** `http://localhost:8501`

---

## 3. Inference Engine & Category Pipeline

The detection pipeline operates in three stages:

1. **Pre-check Compatibility Gate:**
   - Evaluated by `CategoryCompatibilityChecker` (`src/detection/category_checker.py`).
   - Extracts 448D global feature centroid from ResNet-18 layer 2 and layer 3.
   - Computes cosine and Euclidean distance to `models/category_prototypes.pt`.
   - If an incompatible component is uploaded (e.g. Screw image uploaded under Zipper model), halts scanning immediately and triggers the frontend error dialog.

2. **PatchCore v2.3 Feature Matching:**
   - Implemented in `src/detection/patchcore_v23_detector.py` and `src/models/patchcore_v22.py`.
   - Feature extractor maps image to 64×64 spatial patch representations across ResNet-18 `layer1`, `layer2`, and `layer3` (448 channels).
   - Nearest-neighbor Euclidean distance against greedy-coreset subsampled memory bank (`memory_bank.pt`).

3. **Post-Processing & Morphology Localization:**
   - Category-specific calibrated spatial prior (`spatial_prior.pt` / `spatial_prior_p75.pt` for rigid objects like Bottle and Zipper) attenuates nominal edge artifacts.
   - Absolute distance thresholding creates binary anomaly masks.
   - Dual-step morphological closing and opening eliminates isolated sensor noise.
   - Connected component labeling extracts tight defect bounding boxes with area and centroid metrics.

---

## 4. Production Models & Storage Structure

VisionInspect supports five production industrial categories:

| Category | Model Folder | Memory Bank (`memory_bank.pt`) | Spatial Prior | Storage Method |
| :--- | :--- | :--- | :--- | :--- |
| **Bottle** | `models/bottle/patchcore_v23/` | 146.30 MB | `spatial_prior.pt`, `spatial_prior_p75.pt` | Git LFS |
| **Leather** | `models/leather/patchcore_v23/` | 171.50 MB | `spatial_prior.pt` | Git LFS |
| **Transistor** | `models/transistor/patchcore_v23/` | 149.10 MB | `spatial_prior.pt` | Git LFS |
| **Zipper** | `models/zipper/patchcore_v23/` | 168.00 MB | `spatial_prior.pt` | Git LFS |
| **Screw** | `models/screw/patchcore_v23/` | 224.00 MB | `spatial_prior.pt` | Git LFS |
| **Prototypes** | `models/category_prototypes.pt` | 11.0 KB | N/A | Git Standard |

- **Memory banks** are tracked via **Git LFS** in `.gitattributes`.
- **Spatial priors and metadata** (<300 KB each) are tracked directly in Git.
- Verification utility `scripts/verify_models.py` validates all 5 models and detects pointer files.

---

## 5. Required Dependencies

All packages are pinned in `requirements.txt`:
- **Core Deep Learning:** `torch>=2.2.0,<=2.3.0`, `torchvision>=0.17.0,<=0.18.0`
- **Scientific Computing:** `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.2`
- **Vision & Image Processing:** `opencv-python>=4.8.0`, `Pillow>=10.2.0`, `matplotlib>=3.8.0`, `seaborn>=0.13.0`
- **Backend API:** `fastapi>=0.110.0`, `uvicorn>=0.28.0`, `python-multipart>=0.0.9`, `requests>=2.31.0`, `httpx>=0.27.0`
- **Frontend & Reporting:** `streamlit>=1.32.0,<=1.63.0`, `plotly>=5.19.0`, `reportlab>=4.0.0`
- **Configuration & Testing:** `PyYAML==6.0.3`, `pytest>=8.0.0`

---

## 6. Required Environment Variables

All variables have safe local defaults and are templated in `.env.example`:
- `VISIONINSPECT_BACKEND_URL`: URL of the FastAPI backend (Default: `http://127.0.0.1:8000`).
- `KMP_DUPLICATE_LIB_OK`: Set to `TRUE` on Windows to resolve duplicate OpenMP library conflicts.

---

## 7. Runtime Assets vs Development Assets

- **Required for Application Runtime:**
  - `app/` (all frontend and backend entry points, components, and PDF generators)
  - `src/` (core PatchCore model, feature extraction, detection, configuration loaders)
  - `models/` (PatchCore v2.3 category models and category prototypes)
  - `assets/test_samples/` (curated test sample images for offline/demo verification)
  - `config.yaml` (calibrated category thresholds, layer definitions, parameters)
- **Development & Evaluation (not required to run web demo):**
  - `tests/` (109 unit and integration tests)
  - `scripts/` (evaluation, calibration, and training scripts)
  - `dataset/` (external full MVTec AD dataset — excluded from Git; documented in `docs/DATASET_SETUP.md`)

---

## 8. Files Excluded from Git

The `.gitignore` strictly excludes:
- Virtual environments (`.venv/`, `venv/`, `visioninspect_py312/`)
- MVTec raw dataset directories (`dataset/`, `mvtec_anomaly_detection/`)
- Python bytecode and caches (`__pycache__/`, `*.pyc`, `.pytest_cache/`)
- Environment secrets (`.env`, `*.key`, `*.pem`)
- Log dumps and temporary run logs (`*.log`, `task-*.log`)
- Historical experimental models (`models/baseline/`, `autoencoder.pth`, `patchcore_v22`)

---

## 9. Current Known Limitations

1. **CPU Inference Latency:** On standard multi-core CPUs without GPU acceleration, PatchCore nearest-neighbor matching across large coreset banks takes ~0.8s to 2.2s per 256×256 image. With CUDA GPU, latency drops to <45ms.
2. **Category Boundaries:** VisionInspect is calibrated strictly for its 5 production categories (`Bottle`, `Leather`, `Transistor`, `Zipper`, `Screw`). Inspecting unrelated classes will trigger the Category Mismatch warning modal.
