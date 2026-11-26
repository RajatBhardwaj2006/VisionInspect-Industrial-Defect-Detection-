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
   python -m pip install -r requirements.txt
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

Before switching branches, inspect `git status --short` and preserve existing
work. For example, on a clean checkout:

```powershell
git switch main
git pull --ff-only
git switch -c feature/your-feature-name
```

For documentation or mocked API/cache tests, model verification is optional;
production inference tests need usable LFS artifacts and a cached ResNet18
backbone. Do not download weights or run training just to test documentation.

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
python -m pytest tests/ -v
```
Also run the test files you changed independently. See
[Docs/TESTING_GUIDE.md](Docs/TESTING_GUIDE.md) for isolated tests, offline limits,
and interpretation of failures and skips. Report the actual counts and failure
names; do not rely on a historical test-count badge. Investigate existing
failures and explain unavailable prerequisites in the PR rather than hiding them.

## 5. Code style and scope

Follow nearby Python conventions: four-space indentation, `snake_case` functions,
descriptive test names and focused fixtures. Use generated small images and
`tmp_path` for filesystem tests. Mock checkpoint construction or expensive
inference at its boundary, leaving the validation or routing being tested real.
Restore module-level state with pytest's `monkeypatch` so tests remain independent.
No formatter or linter configuration is currently prescribed by the repository.

For documentation/testing contributions, preserve all production behavior:
weights, memory banks, category prototypes, metadata, calibrated thresholds,
feature extraction, inference/scoring, spatial priors, model configuration,
frontend and deployment files. Keep dependency changes separate and discuss them
with maintainers. Improve existing guides instead of duplicating them.

## 6. Pull request checklist

- Review `git diff --check` and `git diff --stat` before staging explicit files.
- Check changed Python syntax and imports in the development environment.
- Check documentation commands and sample filenames against the repository.
- Confirm the diff contains no secrets, personal paths, datasets or generated model artifacts.
- Commit related changes with descriptive `docs:` or `test:` messages.
- Push your branch and open a PR targeting `main`; use a fork if the maintainer
  directs you to that workflow and you lack write access.
- Describe the problem, changed files, new coverage, exact validation results,
  skipped/failed prerequisites and any remaining risks. For docs/tests-only work,
  explicitly confirm production artifacts and logic were unchanged.
- Leave merging to the maintainer. If authentication or repository permissions
  prevent pushing or opening the PR, report the exact error.

Useful references: [runtime architecture](Docs/PROJECT_ARCHITECTURE.md),
[PatchCore explanation](Docs/PATCHCORE_EXPLANATION.md),
[API reference](Docs/API_DOCUMENTATION.md), and
[dataset preparation](Docs/DATASET_SETUP.md).
