"""
VisionInspect - Full Project Structure Setup Script
Run this once inside the folder where VisionInspect/ should exist
(it will use the existing VisionInspect folder if already created - safe to re-run,
won't overwrite or delete anything you already have).

Usage: python setup_project_structure.py
"""

import os

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper",
]

FOLDERS = [
    "dataset/mvtec_anomaly_detection",
    "notebooks",
    "src/data",
    "src/models",
    "src/detection",
    "src/evaluation",
    "src/utils",
    "models",
    "results/predictions",
    "results/heatmaps",
    "results/comparisons",
    "results/metrics",
    "results/reports",
    "app/components",
    "app/assets/icons",
    "app/styles",
    "tests",
    "docs",
]

# One folder per MVTec category inside dataset/ and models/
for cat in CATEGORIES:
    FOLDERS.append(f"dataset/mvtec_anomaly_detection/{cat}")
    FOLDERS.append(f"models/{cat}")

FILES = [
    "README.md",
    "requirements.txt",
    ".gitignore",
    "config.yaml",
    "notebooks/01_dataset_exploration.ipynb",
    "notebooks/02_data_preprocessing.ipynb",
    "notebooks/03_autoencoder_training.ipynb",
    "notebooks/04_anomaly_detection.ipynb",
    "notebooks/05_evaluation.ipynb",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/dataset_loader.py",
    "src/data/preprocessing.py",
    "src/models/__init__.py",
    "src/models/autoencoder.py",
    "src/models/feature_extractor.py",
    "src/detection/__init__.py",
    "src/detection/anomaly_detector.py",
    "src/detection/anomaly_score.py",
    "src/detection/heatmap.py",
    "src/detection/localization.py",
    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/evaluation/roc_curve.py",
    "src/evaluation/visualization.py",
    "src/utils/__init__.py",
    "src/utils/config.py",
    "src/utils/helpers.py",
    "app/app.py",
    "app/components/upload.py",
    "app/components/inspection.py",
    "app/components/heatmap_viewer.py",
    "app/components/batch_results.py",
    "app/components/reports.py",
    "app/styles/style.css",
    "tests/test_dataset.py",
    "tests/test_model.py",
    "tests/test_detection.py",
    "tests/test_app.py",
    "docs/project_report.md",
]

# Per-category model checkpoint placeholders (real .pth files get saved here during training)
for cat in CATEGORIES:
    FILES.append(f"models/{cat}/.gitkeep")

# Binary asset placeholders - real images added later
FILES.append("app/assets/.gitkeep")
FILES.append("app/assets/icons/.gitkeep")
FILES.append("docs/.gitkeep")

ROOT = "VisionInspect"


def main():
    for folder in FOLDERS:
        path = os.path.join(ROOT, folder)
        os.makedirs(path, exist_ok=True)
        print(f"Created folder: {path}")

    for file in FILES:
        path = os.path.join(ROOT, file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as f:
                pass  # empty placeholder, filled in later steps
            print(f"Created file:   {path}")
        else:
            print(f"Already exists: {path} (skipped, not overwritten)")

    print("\nDone. Full VisionInspect/ structure is ready.")
    print("Note: your already-downloaded dataset images inside dataset/mvtec_anomaly_detection/")
    print("were NOT touched or deleted - this script only creates missing folders/files.")


if __name__ == "__main__":
    main()
