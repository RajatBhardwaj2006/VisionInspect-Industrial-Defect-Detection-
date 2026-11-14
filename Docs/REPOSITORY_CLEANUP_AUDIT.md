# VisionInspect — Repository Cleanup & Pre-Submission Audit

**Date:** November 14, 2026  
**Auditor:** Antigravity AI  
**Scope:** Comprehensive pre-submission repository audit and inventory for teacher evaluation.  
**Target Repository:** `VisionInspect — Industrial Defect Detection & Localization`  
**Current Branch:** `chore/pre-submission-repository-cleanup`  
**Safety Backup Branch:** `backup/pre-submission-cleanup-20261114` (pointing to commit `98b5bf2`)  

---

## 1. Executive Summary & Audit Mandate

This audit serves as the formal inventory and evaluation checkpoint before undertaking any repository cleanup.
The primary objective is to ensure that the repository presented for teacher evaluation is:
1. **Professional and clean**: Free from confusing phantom folders, obsolete scaffolding, or misleading artifacts.
2. **100% Functional**: All 5 industrial PatchCore v2.3 models, backend endpoints, Streamlit UI, in-process inference engine, and test suites remain untouched and fully operational.
3. **Reproducible**: Working with a single command from fresh clone on Python 3.12.10.
4. **Safely Documented**: Every file is audited, classified, and justified.

---

## 2. Protected Project Assets (Absolute Freeze)

The following components are **strictly protected** from deletion or modification:
- **ML Models & Weights**:
  - `models/bottle/patchcore_v23/` (Memory bank, metadata, spatial priors)
  - `models/leather/patchcore_v23/` (Memory bank, metadata, spatial priors)
  - `models/transistor/patchcore_v23/` (Memory bank, metadata, spatial priors)
  - `models/zipper/patchcore_v23/` (Memory bank, metadata, spatial priors)
  - `models/screw/patchcore_v23/` (Memory bank, metadata, spatial priors)
  - `models/category_prototypes.pt` (Cross-category mismatch prevention embeddings)
  - `models/baseline/bottle/config.json` & historical metadata (Preserved for baseline comparison tests)
- **Application Core**:
  - `app/app.py` (Streamlit production web application, in-process PatchCore engine, theme management)
  - `app/backend.py` (FastAPI REST service)
  - `app/components/` (`model_docs.py`, `footer.py`)
  - `app/utils/pdf_report.py` (ReportLab PDF generation)
  - `app/assets/icons/` (`visioninspect_3d_logo.png`, `visioninspect_3d_logo_72.png`)
- **Detection & Evaluation Source**:
  - `src/detection/` (PatchCore v2.3 detector, anomaly scoring, heatmap generation, region localization)
  - `src/models/` (Feature extraction with ResNet-18 pyramid pooling)
  - `src/data/` (Dataset loader, preprocessing transforms)
  - `src/evaluation/` (AUROC, pixel metrics, IoU, localization verification)
  - `src/utils/` (Configuration manager, helpers)
- **Deployment & Infrastructure**:
  - `Dockerfile`, `.dockerignore`, `render.yaml`, `packages.txt`, `.python-version`
  - `nginx/nginx.conf.template`
  - `scripts/entrypoint.sh`, `scripts/setup_windows.ps1`, `scripts/start_visioninspect.ps1`
  - `requirements.txt`, `requirements-lock.txt`
  - `.streamlit/config.toml`
- **Standalone Test Samples**:
  - `assets/test_samples/` (Curated test images across all 5 categories allowing offline testing without the 5 GB MVTec dataset)
- **Research Evidence & Results**:
  - `results/` (All locked CSV sweeps, metrics JSONs, diagnostic panels, and evaluation summaries)

---

## 3. Comprehensive File Classification

Every tracked file in the repository has been categorized into one of four tiers:
- **KEEP**: Core production code, model weights, deployment infrastructure, or active documentation.
- **UPDATE**: Critical files requiring updates (e.g., `README.md` for teacher evaluation, `.gitignore` for secret hardening).
- **DELETE**: Confirmed redundant, dead, or misleading scaffolding artifacts.
- **REVIEW**: Files evaluated for necessity and retained due to academic, test, or illustrative value.

### A. Root Configuration & Deployment Files
| File Path | Status | Size | Purpose & Justification |
|:---|:---:|:---:|:---|
| `.devcontainer/devcontainer.json` | KEEP | 1.1 KB | VS Code Dev Container environment specification. |
| `.dockerignore` | KEEP | 320 B | Prevents local caches, datasets, and virtualenvs from entering Docker builds. |
| `.env.example` | KEEP | 640 B | Safe template for optional environment variables without exposing credentials. |
| `.gitattributes` | KEEP | 250 B | Git LFS tracking rules for `.pt` and `.pth` weight files and line ending settings. |
| `.gitignore` | UPDATE | 1.8 KB | Excludes virtualenvs, datasets, bytecode, and caches. Update to include `.streamlit/secrets.toml`. |
| `.python-version` | KEEP | 5 B | Explicitly pins Python runtime to 3.12 for Streamlit Community Cloud. |
| `.streamlit/config.toml` | KEEP | 450 B | Production Streamlit server, dark theme, and telemetry configurations. |
| `CONTRIBUTING.md` | KEEP | 2.5 KB | Academic and open-source contribution guidelines. |
| `Dockerfile` | KEEP | 1.4 KB | Multi-stage production container build for local/cloud container deployment. |
| `LICENSE` | KEEP | 1.1 KB | Standard MIT open-source license. |
| `README.md` | UPDATE | 14.5 KB | Main repository landing document. Update to provide a professional, teacher-ready overview. |
| `config.yaml` | KEEP | 4.8 KB | Central configuration: locked thresholds, spatial priors, morphology parameters for all 5 categories. |
| `inference.py` | KEEP | 4.6 KB | Canonical command-line inference interface for automated testing and scripting. |
| `packages.txt` | KEEP | 20 B | Linux system dependencies (`libgl1`, `libglib2.0-0`) for Streamlit Cloud OpenCV support. |
| `render.yaml` | KEEP | 650 B | Cloud deployment blueprint for Render web service. |
| `requirements-lock.txt` | KEEP | 3.2 KB | Frozen exact package environment lockfile for Python 3.12.10. |
| `requirements.txt` | KEEP | 1.2 KB | Pinned primary dependency manifest for CPU-optimized deployment. |
| `run_manual_inference.ps1` | KEEP | 580 B | Portable PowerShell helper for running CLI inference on test images. |

### B. Application Files (`app/`)
| File Path | Status | Size | Purpose & Justification |
|:---|:---:|:---:|:---|
| `app/__init__.py` | KEEP | 0 B | Python package initializer. |
| `app/app.py` | KEEP | 186 KB | Production Streamlit web application. Contains in-process inference engine, UI cards, and diagnostics. |
| `app/backend.py` | KEEP | 16.5 KB | Production FastAPI service exposing REST endpoints (`/health`, `/predict`, `/categories`). |
| `app/components/__init__.py` | KEEP | 0 B | Component package marker. |
| `app/components/footer.py` | KEEP | 4.2 KB | Standardized responsive footer component. |
| `app/components/model_docs.py` | KEEP | 18.2 KB | Interactive technical documentation tab for all 5 production models. |
| `app/utils/__init__.py` | KEEP | 0 B | Utilities package marker. |
| `app/utils/pdf_report.py` | KEEP | 19.8 KB | ReportLab automated engineering PDF inspection report generator. |
| `app/assets/icons/visioninspect_3d_logo.png` | KEEP | 180 KB | High-resolution brand icon for web app and PDF reports. |
| `app/assets/icons/visioninspect_3d_logo_72.png` | KEEP | 24 KB | Streamlit browser favicon. |

### C. Source Code Files (`src/`)
| Directory / File | Status | Purpose & Justification |
|:---|:---:|:---|
| `src/data/` | KEEP | Dataset loaders, batch processors, and normalization pipelines (`dataset_loader.py`, `preprocessing.py`). |
| `src/detection/` | KEEP | Core detection engines: `patchcore_v23_detector.py` (production), `category_checker.py`, `localization.py`, `heatmap.py`. |
| `src/models/` | KEEP | `feature_extractor.py`, `patchcore_v22.py` (ResNet-18 multi-scale feature representations). |
| `src/evaluation/` | KEEP | `metrics.py`, `pixel_sweep.py`, `roc_curve.py`, `visualization.py`. |
| `src/utils/` | KEEP | `config.py` (YAML configuration loader and model validation), `helpers.py`. |

### D. Model Directory (`models/`)
| File / Directory | Status | Action & Rationale |
|:---|:---:|:---|
| `models/bottle/patchcore_v23/` | KEEP | Production PatchCore v2.3 model weights, memory bank, spatial prior, and metadata. |
| `models/leather/patchcore_v23/` | KEEP | Production PatchCore v2.3 model weights, memory bank, and metadata. |
| `models/screw/patchcore_v23/` | KEEP | Production PatchCore v2.3 model weights, memory bank, and metadata. |
| `models/transistor/patchcore_v23/` | KEEP | Production PatchCore v2.3 model weights, memory bank, and metadata. |
| `models/zipper/patchcore_v23/` | KEEP | Production PatchCore v2.3 model weights, memory bank, and metadata. |
| `models/category_prototypes.pt` | KEEP | Learned prototype embeddings for cross-category compatibility verification. |
| `models/baseline/bottle/config.json` | KEEP | Historical Phase 1 baseline autoencoder configuration. |
| `models/bottle/patchcore/metadata.json` | KEEP | Tested by `tests/test_patchcore.py`. |
| `models/bottle/patchcore_v22/metadata.json` | KEEP | Tested by `tests/test_patchcore_v22.py`. |
| `models/cable/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support cable. 0 bytes. |
| `models/capsule/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support capsule. 0 bytes. |
| `models/carpet/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support carpet. 0 bytes. |
| `models/grid/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support grid. 0 bytes. |
| `models/hazelnut/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support hazelnut. 0 bytes. |
| `models/metal_nut/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support metal_nut. 0 bytes. |
| `models/pill/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support pill. 0 bytes. |
| `models/tile/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support tile. 0 bytes. |
| `models/toothbrush/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support toothbrush. 0 bytes. |
| `models/wood/.gitkeep` | **DELETE** | Phantom scaffolding artifact. VisionInspect does not train or support wood. 0 bytes. |
| `models/bottle/.gitkeep` | KEEP | Retained to maintain directory structure consistency. |
| `models/leather/.gitkeep` | KEEP | Retained to maintain directory structure consistency. |
| `models/screw/.gitkeep` | KEEP | Retained to maintain directory structure consistency. |
| `models/transistor/.gitkeep` | KEEP | Retained to maintain directory structure consistency. |
| `models/zipper/.gitkeep` | KEEP | Retained to maintain directory structure consistency. |

---

## 4. Deletion Plan & Reference Verifications

The following 10 files have been audited for references across the codebase:

```
Checked:
- All Python imports (app/, src/, tests/, scripts/)
- All Configuration files (config.yaml, render.yaml, Dockerfile)
- All Shell/PowerShell scripts (*.sh, *.ps1)
- All Test files (tests/*.py)
- All Documentation files (Docs/*.md)
- GitHub workflows / CI
```

Result: **ZERO active references found.**

### Candidate Deletion Table
| Item # | Target File | Type | Reason for Removal | Verification Test |
|:---:|:---|:---:|:---|:---|
| 1 | `models/cable/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 2 | `models/capsule/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 3 | `models/carpet/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 4 | `models/grid/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 5 | `models/hazelnut/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 6 | `models/metal_nut/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 7 | `models/pill/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 8 | `models/tile/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 9 | `models/toothbrush/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |
| 10 | `models/wood/.gitkeep` | 0-byte placeholder | Phantom scaffolding artifact for unsupported category. | pytest `tests/` |

---

## 5. Execution Protocol (Strict One-at-a-Time Rule)

In strict accordance with the project instructions:
1. One single candidate file will be deleted.
2. Focused validation and/or pytest will be executed.
3. `git status` and `git diff --cached --name-status` will be checked to confirm **exactly one file is staged**.
4. An individual meaningful commit will be recorded (e.g., `chore: remove phantom scaffolding placeholder for cable category`).
5. Only after the commit is successfully recorded will the next file be processed.
6. The full 121-test test suite will be executed before and after the cleanup sequence.

---

## 6. Risk Assessment & Safety Mitigations

- **Risk 1: Accidental Model Deletion**  
  *Mitigation:* All 5 model directories in `models/<category>/patchcore_v23/` are strictly locked and unstageable. Verified with `scripts/verify_models.py`.
- **Risk 2: Deployment Disruption**  
  *Mitigation:* `app/app.py`, `render.yaml`, `Dockerfile`, `.python-version`, and `packages.txt` remain completely untouched during the deletion pass.
- **Risk 3: Git History Corruption**  
  *Mitigation:* Work is conducted on `chore/pre-submission-repository-cleanup`. The safety branch `backup/pre-submission-cleanup-20261114` preserves the exact original commit `98b5bf2`. No force-pushing or history re-writing will ever be performed.
