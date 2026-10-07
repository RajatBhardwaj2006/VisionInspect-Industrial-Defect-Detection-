# VisionInspect — Repository Cleanup & Handoff Audit

> **Document Type:** Maintenance Record & Cleanup Inventory  
> **Date:** October 2026  
> **Status:** Completed & Validated

---

## 1. Codebase Categorization

All files in the repository have been audited into three functional tiers:

### Tier 1: ACTIVE (Runtime & Production)
- `app/` — Streamlit UI, FastAPI backend, PDF reporting, component documentation, footer.
- `src/` — Detection pipeline, PatchCore v2.3 model, feature extractor, category checker, configuration loader.
- `models/` — Production PatchCore v2.3 models for Bottle, Leather, Transistor, Zipper, Screw, plus category prototypes.
- `assets/test_samples/` — Curated, lightweight (17 MB) test sample images for standalone testing without downloading the 5 GB MVTec dataset.
- `scripts/verify_models.py` — Production model integrity and Git LFS pointer validation script.
- `scripts/setup_windows.ps1` — Safe Windows automated environment setup.
- `scripts/start_visioninspect.ps1` — Launcher for backend and frontend services.
- `config.yaml` — Calibrated category thresholds and morphology parameters.
- `requirements.txt` — Core pinned dependencies.

### Tier 2: HISTORICAL (Preserved Academic & Research Evidence)
- `notebooks/` — Exploratory data analysis and initial PatchCore experimentation.
- `results/` — Locked CSV sweeps, per-defect metrics, and phase comparison reports from Phases 1, 2, and 3.
- `scripts/run_multi_category_eval.py`, `scripts/train_patchcore.py` — Benchmark re-execution scripts.
- `docs/` — Historical milestone reports (Phase 1, Phase 2, Phase 3 audits).

### Tier 3: OBSOLETE & REMOVED / EXCLUDED
- `requirements-visioninspect-py312.txt` — Renamed to `requirements-lock.txt` to eliminate ambiguity with `requirements.txt`.
- Unreferenced `.log` files and temporary execution dumps — Purged from git tracking.
- `.pytest_cache/` and `__pycache__/` — Cleared and strictly ignored via `.gitignore`.
- Full raw MVTec dataset — Excluded from Git tracking to maintain repository portability.

---

## 2. Hardcoded Path Elimination

All source code files were audited for developer-specific and machine-specific paths:
- Strictly **zero** hardcoded Windows drive letters (`X:\`, `C:\`) in runtime Python source code.
- Dynamic path resolution via `pathlib.Path(__file__).resolve().parent...` across all modules.
- Portable sample resolver `_resolve_sample()` in `app/app.py` falls back gracefully between `dataset/` and `assets/test_samples/`.
- Verified via automated pytest test assertions in `tests/test_phase3_1_integrity.py` and `tests/test_phase3_2_batch_compatibility.py`.
