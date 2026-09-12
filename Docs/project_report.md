# VisionInspect — Comprehensive Project Overview & Report

- **Author / Maintainer:** [Rajat Bhardwaj](https://github.com/RajatBhardwaj2006)
- **GitHub Repository:** [https://github.com/RajatBhardwaj2006/VisionInspect](https://github.com/RajatBhardwaj2006/VisionInspect)
- **GitHub Profile:** [https://github.com/RajatBhardwaj2006](https://github.com/RajatBhardwaj2006)

---

## Project Overview
VisionInspect is an unsupervised computer vision pipeline for industrial defect detection and localization on MVTec anomaly datasets.

### Model Generations & Milestones
- **Phase 1 (Autoencoder Baseline):** Convolutional Autoencoder (L1 reconstruction loss) on 256x256 images.
- **Phase 2 (PatchCore Feature-Based Anomaly Detector):** ResNet18 feature embeddings with 64x64 patch grid, coreset memory bank, and normal spatial background prior subtraction.
- **Phase 2.3 (Precise Crack Localization):** Direct absolute distance thresholding, crack-preserving morphology, and spatial proximity region merging.

### Verified Benchmark Performance
- **Image AUROC:** `0.9929`
- **Pixel Precision:** `0.7079`
- **Pixel Recall:** `0.6916`
- **Pixel F1-Score:** `0.6996`
- **Pixel IoU:** `0.5380`
