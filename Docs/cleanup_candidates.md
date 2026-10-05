# VisionInspect - Cleanup Candidates Analysis

**Date:** October 5, 2026  
**Auditor:** Antigravity AI  
**Scope:** Evaluation of candidate files for deletion, archival, or cleanup.

---

## Candidate Inventory & Safety Assessment

| FILE | TYPE | WHY IT LOOKS OBSOLETE | REFERENCES FOUND | SAFE TO DELETE? | CONFIDENCE | DEPENDENCIES |
|:---|:---|:---|:---|:---|:---|:---|
| `results/Updated_model_predictions.zip` | Redundant Zip Archive (425.5 MB) | Exact duplicate zip of `results/Updated_model_predictions/`. Exceeds GitHub's 100MB file limit. Uncompressed folder is preserved. | None in code or tests. | **YES (DELETE)** | 100% | None |
| `scratch/` (entire directory, 13 files, 8.1 MB) | Temporary Scratch Code & Cache | One-off exploratory scripts and pickle files from screw debugging (`val_summary.pkl`, `calibrate_screw_validation.py`, etc.). | None in production, backend, frontend, or tests. Untracked in git. | **YES (DELETE)** | 100% | None |
| `notebooks/02_data_preprocessing.ipynb` | Jupyter Notebook (0 bytes) | Empty placeholder file generated during initial scaffolding. Contains no code. | `setup_project_structure.py` | **YES (DELETE)** | 100% | None |
| `notebooks/05_evaluation.ipynb` | Jupyter Notebook (0 bytes) | Empty placeholder file generated during initial scaffolding. Contains no code. | `setup_project_structure.py` | **YES (DELETE)** | 100% | None |
| `docs/walkthrough/project_report.md` | Markdown File (0 bytes) | Empty placeholder in abandoned subfolder. | None. | **YES (DELETE)** | 100% | None |
| `PHASE_3_PRE_IMPLEMENTATION_AUDIT.md` | Markdown Document (39 KB) | Historical Phase 3 audit from Sep 10, 2026 cluttering root directory. `docs/phase3_pre_implementation_audit.md` is a stub pointing to it. | Referenced by `docs/phase3_pre_implementation_audit.md`. | **SAFE TO CONSOLIDATE INTO DOCS** | 100% | Move content into `docs/phase3_pre_implementation_audit.md` |
| `CLEANUP_REPORT.md` | Markdown Document (7.4 KB) | Historical cleanup report from Sep 25, 2026 cluttering root directory. | None in code/tests. | **SAFE TO ARCHIVE INTO DOCS** | 100% | Move to `docs/archive/cleanup_report_2026_09_25.md` |
| `setup_project_structure.py` | Python Script (3.5 KB) | One-time initial project scaffolding utility. Project directory is established. | Referenced in historical markdown docs. | **SAFE TO MOVE TO SCRIPTS** | 100% | Move to `scripts/setup_project_structure.py` |
| `run_manual_inference.ps1` | PowerShell Script (0.7 KB) | Contains hardcoded Windows virtualenv path (`X:\VScode\...`). | None in code/tests. | **KEEP & REFACTOR** (Make path portable) | 100% | `inference.py` |
| `models/baseline/bottle/autoencoder.pth` | PyTorch Weights (5.3 MB) | Baseline autoencoder model from Phase 1. | Referenced in reports; baseline comparison. | **KEEP / PROTECT** | 100% | Baseline documentation |
| `models/bottle/autoencoder.pth` | PyTorch Weights (5.3 MB) | Autoencoder weights. | **Directly imported in `tests/test_model.py`** | **KEEP / PROTECT (DO NOT DELETE)** | 100% | `tests/test_model.py` |
| `models/bottle/patchcore/memory_bank.pt` | Memory Bank (31.4 MB) | PatchCore v2.0 memory bank. | **Directly imported in `tests/test_patchcore.py`** | **KEEP / PROTECT (DO NOT DELETE)** | 100% | `tests/test_patchcore.py` |
| `models/bottle/patchcore_v22/memory_bank.pt` | Memory Bank (146.3 MB) | PatchCore v2.2 memory bank. | **Directly imported in `tests/test_patchcore_v22.py`** and `test_model_dispatch.py` | **KEEP / PROTECT (DO NOT DELETE)** | 100% | `tests/test_patchcore_v22.py` |
| `models/<all 5 categories>/patchcore_v23/` | Production Weights & Memory Banks | Current production model directories. | **Production backend, inference CLI, Streamlit app, test suite** | **STRICTLY PROTECTED (DO NOT TOUCH)** | 100% | Production inference |
| `results/Updated_model_predictions/` | Final Evaluation Directory | Latest audited predictions (530 files, 432 MB). | Final audit report and per-sample visual evidence. | **STRICTLY PROTECTED (DO NOT TOUCH)** | 100% | Production audit documentation |
| `results/predictions/`, `comparisons/`, `*_diagnostic/` | Temporary Evaluation Dumps | Runtime diagnostic image dumps. | None (already git-ignored). | **KEEP LOCAL / GIT-IGNORED** | 100% | Local diagnostics |

---

## Action Plan Summary

1. **Delete Confirmed Unused & Redundant Artifacts:**
   - Remove `results/Updated_model_predictions.zip` (frees 425 MB, prevents GitHub push block).
   - Clean `scratch/` directory and ensure it is ignored in `.gitignore`.
   - Remove 0-byte placeholder files: `notebooks/02_data_preprocessing.ipynb`, `notebooks/05_evaluation.ipynb`, `docs/walkthrough/project_report.md`.
2. **Reorganize Root Files for Professional Presentation:**
   - Consolidate `PHASE_3_PRE_IMPLEMENTATION_AUDIT.md` into `docs/phase3_pre_implementation_audit.md`.
   - Move `CLEANUP_REPORT.md` into `docs/archive/cleanup_report_2026_09_25.md`.
   - Move `setup_project_structure.py` into `scripts/setup_project_structure.py`.
3. **Fix Encodings and Portability:**
   - Strip UTF-8 BOM from `tests/test_model_dispatch.py`.
   - Fix hardcoded paths in `run_manual_inference.ps1` and `src/detection/anomaly_detector.py`.
   - Clean and unify `requirements.txt` into modern UTF-8 format.
4. **Synchronize Test Assertions:**
   - Update `tests/test_phase3_3_transistor.py` and `tests/test_phase3_multi_category.py` to assert active validated constants (Screw spatial prior False, Leather threshold 2.90, Screw threshold 2.80).
