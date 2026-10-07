# Contributing to VisionInspect

Welcome to the VisionInspect project! This guide provides simple rules for team members working on the codebase.

---

## 1. Getting Started

1. Clone the repository with Git LFS:
   ```bash
   git clone https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git
   cd VisionInspect-Industrial-Defect-Detection-
   git lfs pull
   ```
2. Set up your Python 3.12 environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. Verify models:
   ```powershell
   python scripts/verify_models.py
   ```

---

## 2. Branch Naming & Workflow

- Feature branch: `feature/your-feature-name`
- Bugfix branch: `fix/issue-description`
- Documentation: `docs/topic-name`

Always branch off `main` and submit a Pull Request.

---

## 3. Commit Expectations & Hygiene

- Write clear, descriptive commit messages:
  - `feat: add model documentation page`
  - `fix: resolve pdf character escaping`
  - `docs: update deployment instructions`
- **Never commit credentials, secrets, or `.env` files.**
- **Never commit raw dataset dumps** (`dataset/mvtec_anomaly_detection/`).
- **Never modify locked production model weights** (`models/*/patchcore_v23/`) without consulting the team.
- **Never hardcode machine-specific paths** (`X:\...`, `C:\Users\YourName\...`). Always use `pathlib.Path` relative to the project root.

---

## 4. Testing Before Pushing

Always run the full test suite before committing or pushing changes:
```powershell
pytest tests/
```
All 109+ tests must pass before opening a Pull Request.
