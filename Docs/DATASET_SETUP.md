# VisionInspect — MVTec AD Dataset Setup Guide

> **Notice:** The full MVTec AD dataset is **not required** to run VisionInspect, use the Streamlit application, test the API, or run unit tests. Pre-trained production models are included in `models/`, and curated test images are provided in `assets/test_samples/`.  
> This guide is strictly for developers who wish to **re-train models**, re-run full benchmark sweeps, or evaluate on the complete multi-gigabyte MVTec test sets.

---

## 1. Dataset Overview

VisionInspect is developed using the **MVTec Anomaly Detection (MVTec AD)** benchmark dataset:
- **Reference:** Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
- **Official Download Portal:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 2. Supported Categories

VisionInspect calibrates and evaluates on 5 industrial categories:
1. **Bottle** (Rigid glass object — surface cracks, chipping, contamination)
2. **Leather** (Surface texture — cuts, color marks, folds, poke flaws)
3. **Transistor** (Semiconductor component — bent leads, cut pins, damaged casing, missing body)
4. **Zipper** (Mechanical fastener — broken teeth, split gaps, fabric edge roughness)
5. **Screw** (Threaded metal fastener — thread deformation, drive head scratches, manipulated faces)

---

## 3. Expected Directory Structure

When downloading and extracting MVTec AD archives, place the files under `dataset/mvtec_anomaly_detection/` at the project root:

```
VisionInspect/
└── dataset/
    └── mvtec_anomaly_detection/
        ├── bottle/
        │   ├── train/
        │   │   └── good/             # Normal training images (.png)
        │   ├── test/
        │   │   ├── good/             # Normal test images (.png)
        │   │   ├── broken_large/     # Defect sample images (.png)
        │   │   ├── broken_small/
        │   │   └── contamination/
        │   └── ground_truth/         # Pixel-level binary masks (.png)
        ├── leather/
        │   ├── train/good/
        │   ├── test/
        │   └── ground_truth/
        ├── screw/
        ├── transistor/
        └── zipper/
```

---

## 4. How Scripts Locate the Dataset

Evaluation and training scripts in `scripts/` configure the dataset path via command-line arguments and configuration:

```powershell
# Retrain a specific category:
python scripts/train_patchcore.py --category bottle --dataset_root dataset/mvtec_anomaly_detection

# Run evaluation sweep across categories:
python scripts/run_multi_category_eval.py --dataset_root dataset/mvtec_anomaly_detection
```

Default paths are set in `config.yaml` and script defaults to `dataset/mvtec_anomaly_detection`.

---

## 5. Git Status

The `dataset/` directory is explicitly excluded in `.gitignore` to prevent multi-gigabyte binaries from cluttering the repository.
