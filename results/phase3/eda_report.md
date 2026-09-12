# MVTec Anomaly Detection — Phase 3.0 Dataset EDA Report

This report details the dataset distributions, image dimensions, defect breakdowns, and ground-truth mask pixel statistics across the 5 initial categories in VisionInspect Phase 3.0.

## 1. Summary Overview

| Category | Train Normal | Test Normal | Test Defective | Total Test | Defect Types | Resolution | Mean Defect Area | Max Defect Area |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** | 209 | 20 | 63 | 83 | 3 | 900x900 | 7.62% | 27.42% |
| **leather** | 245 | 32 | 92 | 124 | 5 | 1024x1024 | 0.87% | 6.86% |
| **transistor** | 213 | 60 | 40 | 100 | 4 | 1024x1024 | 11.98% | 49.32% |
| **zipper** | 240 | 32 | 119 | 151 | 7 | 1024x1024 | 2.62% | 11.83% |
| **screw** | 320 | 41 | 119 | 160 | 5 | 1024x1024 | 0.34% | 1.01% |

## 2. Category Detail Breakdown

### Category: `bottle`

- **Resolution**: 900x900
- **Train Normal Images**: 209
- **Test Normal Images**: 20
- **Test Defective Images**: 63
- **Mean Defect Area**: 61694.1 px (7.62% of image)
- **Max Defect Area**: 222091 px (27.42% of image)

#### Defect Type Distribution

| Defect Type | Test Count | Mean Defect Area (px) | Mean Area (%) |
|---|:---:|:---:|:---:|
| `broken_large` | 20 | 94789.8 px | 11.70% |
| `broken_small` | 22 | 24852.8 px | 3.07% |
| `contamination` | 21 | 68770.1 px | 8.49% |
| `good` (normal) | 20 | - | - |

---

### Category: `leather`

- **Resolution**: 1024x1024
- **Train Normal Images**: 245
- **Test Normal Images**: 32
- **Test Defective Images**: 92
- **Mean Defect Area**: 9166.9 px (0.87% of image)
- **Max Defect Area**: 71926 px (6.86% of image)

#### Defect Type Distribution

| Defect Type | Test Count | Mean Defect Area (px) | Mean Area (%) |
|---|:---:|:---:|:---:|
| `color` | 19 | 4385.9 px | 0.42% |
| `cut` | 19 | 5232.4 px | 0.50% |
| `fold` | 17 | 26327.4 px | 2.51% |
| `glue` | 19 | 8692.2 px | 0.83% |
| `good` (normal) | 32 | - | - |
| `poke` | 18 | 2660.3 px | 0.25% |

---

### Category: `transistor`

- **Resolution**: 1024x1024
- **Train Normal Images**: 213
- **Test Normal Images**: 60
- **Test Defective Images**: 40
- **Mean Defect Area**: 125650.8 px (11.98% of image)
- **Max Defect Area**: 517163 px (49.32% of image)

#### Defect Type Distribution

| Defect Type | Test Count | Mean Defect Area (px) | Mean Area (%) |
|---|:---:|:---:|:---:|
| `bent_lead` | 10 | 16773.1 px | 1.60% |
| `cut_lead` | 10 | 17185.5 px | 1.64% |
| `damaged_case` | 10 | 29866.6 px | 2.85% |
| `good` (normal) | 60 | - | - |
| `misplaced` | 10 | 438778.1 px | 41.85% |

---

### Category: `zipper`

- **Resolution**: 1024x1024
- **Train Normal Images**: 240
- **Test Normal Images**: 32
- **Test Defective Images**: 119
- **Mean Defect Area**: 27515.2 px (2.62% of image)
- **Max Defect Area**: 124057 px (11.83% of image)

#### Defect Type Distribution

| Defect Type | Test Count | Mean Defect Area (px) | Mean Area (%) |
|---|:---:|:---:|:---:|
| `broken_teeth` | 19 | 17468.6 px | 1.67% |
| `combined` | 16 | 43600.6 px | 4.16% |
| `fabric_border` | 17 | 17666.0 px | 1.68% |
| `fabric_interior` | 16 | 19290.0 px | 1.84% |
| `good` (normal) | 32 | - | - |
| `rough` | 17 | 46308.4 px | 4.42% |
| `split_teeth` | 18 | 31763.4 px | 3.03% |
| `squeezed_teeth` | 16 | 17303.2 px | 1.65% |

---

### Category: `screw`

- **Resolution**: 1024x1024
- **Train Normal Images**: 320
- **Test Normal Images**: 41
- **Test Defective Images**: 119
- **Mean Defect Area**: 3517.7 px (0.34% of image)
- **Max Defect Area**: 10599 px (1.01% of image)

#### Defect Type Distribution

| Defect Type | Test Count | Mean Defect Area (px) | Mean Area (%) |
|---|:---:|:---:|:---:|
| `good` (normal) | 41 | - | - |
| `manipulated_front` | 24 | 3261.8 px | 0.31% |
| `scratch_head` | 24 | 2597.5 px | 0.25% |
| `scratch_neck` | 25 | 3476.6 px | 0.33% |
| `thread_side` | 23 | 2933.6 px | 0.28% |
| `thread_top` | 23 | 5374.0 px | 0.51% |

---

## 3. Engineering Insights for Multi-Category Detection

1. **Structural Objects vs. Textures**:
   - `leather` is a pure unaligned texture. Normal samples lack consistent spatial landmarks, making normal spatial priors counterproductive. `use_spatial_prior: false` enables uniform patch feature matching across texture surfaces.
   - `bottle`, `transistor`, `zipper`, and `screw` are structured objects with rigid geometry and background boundaries, benefiting significantly from spatial prior subtraction.
2. **Defect Scale Disparity**:
   - Defect areas vary widely: from small pinhole punctures and thread scratches (<0.5% image area) to large broken portions (>10% image area).
   - High-resolution feature extraction (64x64 grid, 4x4 px patch receptive field) is vital to preserve tiny defect signatures without spatial dilution.
3. **Category-Specific Distance Baselines**:
   - Feature distance norms differ fundamentally by product surface complexity. Thresholds must be calibrated per category rather than applied globally.
