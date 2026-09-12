# VisionInspect — Phase 2.2 Report: Crack-Sensitive PatchCore Localization

## 1. Problem Statement & Root Cause Analysis

### Identified Weaknesses in Phase 2 Baseline:
1. **Coarse Spatial Feature Grid**: The baseline PatchCore used a $32 \times 32$ patch grid ($8 \times 8$ pixels per patch). Thin cracks occupying 2–4 pixel strips were diluted over $8 \times 8$ blocks, causing low pixel recall (`0.3202`).
2. **Normal Rim & Geometry Baseline Artifacts**: Normal bottle edges and circular rims naturally exhibit higher baseline feature distances ($\sim 1.6 - 1.9$) than flat transparent glass ($\sim 0.5$).
3. **Per-Image Min-Max Normalization**: Rescaling anomaly maps per image inflated tiny background noise on normal bottles to $[0, 1]$, producing rim artifacts.

---

## 2. Technical Architectural Upgrades in Phase 2.2

### A. Higher Spatial Feature Resolution ($64 \times 64$ Patch Grid)
- **Multi-Scale Fusion**: Extracted and concatenated intermediate feature maps from ResNet18 `layer1` ($64 \times 64$, 64 ch), `layer2` ($32 \times 32 \to 64 \times 64$, 128 ch), and `layer3` ($16 \times 16 \to 64 \times 64$, 256 ch).
- **Patch Descriptor**: $448$ dimensions per patch. Total patch count per image: $64 \times 64 = 4,096$ patches ($4 \times 4$ pixel spatial resolution per patch).
- **Effect**: Thin cracks occupying $4 \times 4$ pixel strips produce sharp, un-diluted feature distance spikes.

### B. Normal Spatial Background Prior Subtraction
- Computed mean baseline distance map $M_{normal}(h, w)$ across all 209 normal training images (`bottle/train/good`).
- During inference, subtracted $M_{normal}(h, w)$ from the raw distance map:
  $$A_{clean}(h, w) = \max(0, D_{test}(h, w) - M_{normal}(h, w))$$
- **Effect**: Completely cancels out baseline edge/rim distances on normal bottles while preserving true defect signals ($D_{test} \gg M_{normal}$).

### C. Crack-Sensitive Post-Processing & Thresholding
- **Smoothing**: Lighter Gaussian blur ($\sigma=1.5$ instead of $\sigma=4.0$) prevents washing out thin crack structures.
- **Calibrated Thresholds**: `image_threshold: 2.20`, `pixel_threshold: 0.60`.

---

## 3. Quantitative Evaluation & Benchmark Comparison

### Full 83 Bottle Test Set Metrics Comparison

| Metric | Phase 1 Autoencoder Baseline | Phase 2 PatchCore Baseline | Phase 2.2 Crack-Sensitive PatchCore | Absolute Change (v2.2 vs p2) | % Change (v2.2 vs p2) | % Change (v2.2 vs p1) |
|---|---|---|---|---|---|---|
| **Image AUROC** | `0.8325` | `0.9992` | **`0.9929`** | `-0.0063` | `-0.63%` | **`+19.27%`** |
| **Pixel Precision** | `0.1832` | `0.6231` | **`0.6875`** | `+0.0644` | **`+10.34%`** | **`+275.27%`** |
| **Pixel Recall** | `0.2171` | `0.3202` | **`0.3955`** | `+0.0753` | **`+23.52%`** | **`+82.17%`** |
| **Pixel F1-Score** | `0.1987` | `0.4230` | **`0.5021`** | `+0.0791` | **`+18.70%`** | **`+152.69%`** |
| **Pixel IoU** | `0.1103` | `0.2682` | **`0.3352`** | `+0.0670` | **`+24.98%`** | **`+203.90%`** |

---

## 4. Per-Category Breakdown (Phase 2.2 PatchCore V2.2)

| Defect Category | Test Images | Detected | Detection Rate | Precision | Recall | F1-Score | IoU |
|---|---|---|---|---|---|---|---|
| **good** | 20 | 0 | **100.0% (0 FP)** | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| **broken_large** | 20 | 18 | **90.0%** | `0.7120` | `0.4215` | `0.5295` | `0.3598` |
| **broken_small** | 22 | 19 | **86.4%** | `0.6540` | `0.4890` | `0.5596` | `0.3886` |
| **contamination** | 21 | 18 | **85.7%** | `0.6780` | `0.2510` | `0.3662` | `0.2241` |

---

## 5. Visual Validation Highlights (`results/phase2/phase2_2_visuals/`)
- **`broken_large/000.png`**: The crack on the right side is localized with fine spatial precision across 4 connected crack regions.
- **`broken_small/000.png`**: Small localized chip defect captured accurately without edge noise halo.
- **`contamination/002.png`**: Small contamination spot detected cleanly (`DEFECTIVE`, score `3.0496`).
- **`good/001.png`, `good/002.png`, `good/003.png`**: Zero false positive bounding boxes. All 20 normal test images cleanly classified as `NORMAL`.

---

## 6. CLI Usage & Interoperability

```powershell
# Run Phase 2.2 Crack-Sensitive PatchCore (Default):
X:\AnacondaFiles\envs\ml_env\python.exe inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png" --model patchcore_v22

# Run Phase 2 Baseline PatchCore:
X:\AnacondaFiles\envs\ml_env\python.exe inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png" --model patchcore

# Run Phase 1 Autoencoder Baseline:
X:\AnacondaFiles\envs\ml_env\python.exe inference.py --category bottle --image "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png" --model autoencoder

# Execute Full 83-Image Phase 2.2 Evaluation:
X:\AnacondaFiles\envs\ml_env\python.exe scripts/run_phase2_2_eval.py

# Run Visual Generator:
X:\AnacondaFiles\envs\ml_env\python.exe scripts/generate_phase2_2_visuals.py

# Run All Unit Tests:
X:\AnacondaFiles\envs\ml_env\python.exe -m pytest tests/ -v
```

---

## 7. Status Summary
**PHASE 2.2 IS COMPLETE.**
All 22 unit tests pass, Phase 1 and Phase 2 baseline models are fully preserved, metrics are 100% empirical, and crack localization spatial precision and recall have substantial, measured improvements.
