# Transistor Performance Comparison: Phase 3.1 → Phase 3.2 → Phase 3.3

| Metric | Phase 3.1 Baseline | Phase 3.2 Localization | Phase 3.3 Targeted (Locked) | Absolute Change (vs 3.2) |
|---|---|---|---|---|
| **AUROC** | 0.8200 | 0.8200 | **0.9154** | **+0.0954** |
| **Detection Rate** | 15.0% (6/40) | 30.0% (12/40) | **80.0% (32/40)** | **+50.0%** |
| **Normal Accuracy** | 100.0% (60/60) | 93.3% (56/60) | **91.7% (55/60)** | -1.6% |
| **Pixel Precision** | 0.0000 | 0.5323 | **0.5857** | **+0.0534** |
| **Pixel Recall** | 0.0000 | 0.0442 | **0.2744** | **+0.2302** (6.2x) |
| **Pixel F1** | 0.0000 | 0.0816 | **0.3737** | **+0.2921** (4.6x) |
| **Pixel IoU** | 0.0000 | 0.0425 | **0.2298** | **+0.1873** (5.4x) |

### Per-Defect Detection Comparison

| Defect Type | Total Samples | Phase 3.2 Detected | Phase 3.3 Detected | Phase 3.3 Detection Rate |
|---|---|---|---|---|
| `bent_lead` | 10 | 9/10 | **10/10** | **100.0%** |
| `cut_lead` | 10 | 0/10 | **9/10** | **90.0%** |
| `damaged_case` | 10 | 0/10 | **5/10** | **50.0%** |
| `misplaced` | 10 | 3/10 | **8/10** | **80.0%** |
| `good` (Normals) | 60 | 4 FP | **5 FP** | **91.7%** |
