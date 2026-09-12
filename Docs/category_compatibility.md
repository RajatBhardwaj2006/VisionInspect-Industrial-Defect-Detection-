# VisionInspect — Category Compatibility Checking & Misconfiguration Prevention

**Date:** September 2026  
**Module:** `src/detection/category_checker.py`  
**API Endpoints:** `POST /predict`, `POST /predict/batch`  
**Maintainer:** [Rajat Bhardwaj](https://github.com/RajatBhardwaj2006)  

---

## 1. Executive Summary

In industrial defect inspection systems, deploying category-specific models introduces a critical operational hazard: **category misconfiguration**. If an operator or automated conveyor feeds a product into an incorrect category model (for example, inspecting a metal `zipper` using a model calibrated on glass `bottle` normal representations), the system produces erroneous results. 

VisionInspect Phase 3.2 introduces a lightweight, conservative **Category Compatibility Checker** designed to detect when an uploaded image appears markedly more compatible with a different supported category.

> [!IMPORTANT]
> **Safety Design Principle:** The Category Compatibility Checker is **NOT** a product classifier. It does not attempt to categorize arbitrary objects in the wild. It functions strictly as an operational sanity check across supported categories and **NEVER** automatically alters the user's selected model without explicit user confirmation.

---

## 2. The Danger of Wrong-Category Inference

PatchCore v2.3 detects defects by identifying nearest-neighbor distances between an input image's patch representations and a memory bank of nominal product patches:

1. **Massive False Alarms:** If a `zipper` is passed into the `bottle` model, normal zipper teeth and fabric texture are completely absent from the bottle memory bank. The entire zipper is flagged as anomalous with maximum scores, masking any genuine defects.
2. **Severely Distorted Anomaly Maps:** Normal structural features generate noisy background heatmaps, obscuring crack detection and localized defect bounding boxes.
3. **Threshold Mismatch:** Decision thresholds are calibrated per category (e.g. `1.50` for bottle vs `2.28` for screw). Evaluating an image against uncalibrated thresholds produces invalid operational decisions.

---

## 3. Mathematical Formulation

### 3.1 Feature Representation
Each image $I$ is passed through the frozen ResNet18 multi-scale extractor (Layers 1, 2, and 3), producing a 64x64 grid of 448-dimensional patch representations:

$$\mathbf{P}(I) \in \mathbb{R}^{4096 \times 448}$$

The global visual signature of the image is computed as the mean patch descriptor:

$$\bar{\mathbf{p}} = \frac{1}{4096} \sum_{i=1}^{4096} \mathbf{p}_i \in \mathbb{R}^{448}$$

### 3.2 Category Prototypes
For each supported category $c \in \{\text{bottle}, \text{leather}, \text{transistor}, \text{zipper}, \text{screw}\}$, the category prototype $\mathbf{c}_k \in \mathbb{R}^{448}$ represents the centroid of all normal training patches:

$$\mathbf{c}_k = \frac{1}{M_k} \sum_{m=1}^{M_k} \mathbf{v}_m^{(k)}$$

where $\mathcal{M}_k = \{\mathbf{v}_m^{(k)}\}_{m=1}^{M_k}$ is the coreset memory bank of category $k$. These prototypes are precomputed and cached in `models/category_prototypes.pt` (< 9 KB footprint).

### 3.3 Distance & Confidence Margin
For an input image with descriptor $\bar{\mathbf{p}}$ and selected category $c_{\text{sel}}$, Euclidean distances are computed to all category prototypes:

$$d_k = \|\bar{\mathbf{p}} - \mathbf{c}_k\|_2 \quad \forall k \in \mathcal{C}$$

The most compatible category is defined as:

$$k^* = \arg\min_{k \in \mathcal{C}} d_k$$

The **confidence margin** $\Delta$ measures how much closer the best matching category is compared to the user's selected category:

$$\Delta = d_{c_{\text{sel}}} - d_{k^*}$$

---

## 4. Conservative Decision Rules

To avoid nuisance alarms and prevent false warnings on genuine, heavily damaged defects, the checker enforces strict conservative criteria:

$$\text{Decision} = \begin{cases} 
\text{COMPATIBLE} & \text{if } k^* = c_{\text{sel}} \\
\text{MISMATCH WARNING} & \text{if } k^* \neq c_{\text{sel}} \text{ and } \Delta \ge \tau_{\text{margin}} \\
\text{INCONCLUSIVE} & \text{if } k^* \neq c_{\text{sel}} \text{ and } \Delta < \tau_{\text{margin}}
\end{cases}$$

Where the default margin threshold is calibrated at:

$$\tau_{\text{margin}} = 0.35$$

### Empirical Margin Separation:
- **Same category (normal product):** $\Delta = 0.00$, distance $\le 0.45$.
- **Same category (severely defective product):** $k^* = c_{\text{sel}}$, distance $\approx 0.40 - 0.55$, $\Delta = 0.00$. Zero false warnings.
- **Cross-category mismatch (e.g. zipper under bottle):** $k^* = \text{zipper}$, $d_{\text{bottle}} \approx 1.50$, $d_{\text{zipper}} \approx 0.45$, $\Delta \approx 1.05 \gg 0.35$. Strong, confident warning generated.
- **Ambiguous input (e.g. blank background or synthetic solid pattern):** $\Delta < 0.35$. Marked as inconclusive, preventing spurious warnings.

---

## 5. User Autonomy & Workflow Integration

### 5.1 No Automatic Switching Principle
Automatic category switching in industrial pipelines is inherently dangerous. For instance, if an unexpected surface contamination resembles another texture, automatically changing the model would corrupt compliance logging.

Therefore:
1. The backend **always processes the image using the operator's explicitly requested model**.
2. The warning is returned in the API payload:
   ```json
   "category_compatibility": {
     "selected_category": "bottle",
     "best_compatible_category": "zipper",
     "is_mismatch": true,
     "confidence_margin": 1.052,
     "warning_message": "Selected category: Bottle. This image appears more compatible with: Zipper."
   }
   ```
3. In the Streamlit UI, a prominent advisory box displays:
   - `⚠️ CATEGORY MISMATCH WARNING`
   - `Selected Category: Bottle | Detected Appearance: Zipper (Confidence margin: 1.05)`
   - Two distinct buttons: `[Switch to Zipper]` and `[Continue with Bottle]`.

---

## 6. Computational Efficiency

| Metric | Measured Value | Target Budget |
|:---|:---:|:---:|
| Storage Footprint | 8.9 KB (`models/category_prototypes.pt`) | < 1 MB |
| RAM Footprint | ~9 KB (CPU/GPU tensor) | < 5 MB |
| Inference Latency Overhead | 2.8 ms (CPU) / 0.9 ms (CUDA) | < 15 ms |
| Additional Memory Banks Loaded | 0 (only active category model loaded) | 0 |

The compatibility check adds virtually zero runtime overhead because the patch representations are computed via the same ResNet18 backbone used for PatchCore defect localization.
