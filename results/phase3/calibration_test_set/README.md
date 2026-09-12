# HISTORICAL TEST-SET CALIBRATION — NOT UNBIASED FINAL BENCHMARK

> [!WARNING]
> **Data Leakage Notice**: The evaluation metrics stored in this folder (`results/phase3/calibration_test_set/`) were obtained in Phase 3.0 by optimizing the 2D threshold grid sweep directly against the final MVTec test set ground-truth annotations.
>
> While useful for establishing the theoretical ceiling of the PatchCore v2.3 architecture with post-hoc tuned thresholds, this methodology peeks at the test labels during calibration.
>
> In Phase 3.1, a rigorous **Train -> Normal Validation Calibration -> Final Locked Test** methodology was established.
> The new unbiased evaluation benchmark is stored separately in:
> - `results/phase3/final_validation_locked/` (validation calibration metrics & locked parameters)
> - `results/phase3/final_test/` (final unbiased test set evaluation)

### Historical Metrics Summary (Test-Calibrated)
- **bottle**: AUROC 0.9929, F1 0.6996, IoU 0.5380 (img_th=1.50, pixel_th=1.40)
- **leather**: AUROC 0.9236, F1 0.5307, IoU 0.3612 (img_th=2.20, pixel_th=2.71)
- **transistor**: AUROC 0.8200, F1 0.4113, IoU 0.2589 (img_th=2.69, pixel_th=0.88)
- **zipper**: AUROC 0.8981, F1 0.5732, IoU 0.4018 (img_th=1.47, pixel_th=0.91)
- **screw**: AUROC 0.8776, F1 0.4884, IoU 0.3231 (img_th=2.28, pixel_th=2.00)
