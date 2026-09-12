# VisionInspect — Industrial Defect Detection & Localization

[![Maintainer](https://img.shields.io/badge/Maintainer-RajatBhardwaj2006-blue.svg)](https://github.com/RajatBhardwaj2006)
[![GitHub Repository](https://img.shields.io/badge/GitHub-VisionInspect-green.svg)](https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.11-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-orange.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-35%2F35%20Passing-success.svg)](tests/)

VisionInspect is an unsupervised industrial anomaly detection and defect localization system built for high-precision manufacturing quality assurance. It leverages deep feature representations from a ResNet18 backbone, high-resolution patch-level memory banks, adaptive spatial prior modeling, and precise morphology-based localization.

- **Author / Maintainer:** [Rajat Bhardwaj](https://github.com/RajatBhardwaj2006)
- **GitHub Repository:** [VisionInspect-Industrial-Defect-Detection-](https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git)
- **Dataset:** MVTec Anomaly Detection (MVTec AD) Benchmark

---

## Key Features

- **Unsupervised Learning Paradigm**: Trains purely on normal (good) product images; requires zero defect annotations during training.
- **High-Resolution PatchCore v2.3 Architecture**: Combines multi-scale features from layers 1, 2, and 3 into 448-dimensional patch representations over a 64x64 grid (4x4 px receptive field).
- **Adaptive Spatial Prior Subtraction**: Removes repetitive normal background and edge artifacts for rigid geometric objects while automatically disabling prior subtraction for unaligned textural surfaces (e.g., leather).
- **Multi-Category Framework (Phase 3.0)**:
  - Initial 5 supported categories: `bottle`, `leather`, `transistor`, `zipper`, `screw`.
  - Dynamic model path resolution and runtime category integrity validation.
  - Empirically calibrated 2D threshold optimization per category.
- **Production-Ready Web Stack**:
  - **FastAPI Backend (`app/backend.py`)**: RESTful inference API with `/health`, `/categories`, and `/predict` endpoints.
  - **Streamlit Web UI (`app/app.py`)**: Interactive inspection dashboard with sample selector, live upload, anomaly heatmaps, and bounding box tables.

---

## Performance Benchmarks

All models were evaluated on the full test datasets using the ResNet18 PatchCore v2.3 architecture:

| Category | Product Type | Test Images | Image AUROC | Pixel Precision | Pixel Recall | Pixel F1 | Pixel IoU | Detection Rate | Normal Accuracy | Calibrated Img Thresh | Calibrated Pixel Thresh |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bottle** *(Baseline)* | Rigid Object | 83 | **0.9929** | **0.7079** | **0.6915** | **0.6996** | **0.5380** | 95.2% | 70.0% | 1.50 | 1.40 |
| **leather** | Surface Texture | 124 | **0.9236** | **0.5366** | **0.5249** | **0.5307** | **0.3612** | 72.8% | **100.0%** | 2.20 | 2.71 |
| **zipper** | Periodic Object | 151 | **0.8981** | **0.5152** | **0.6460** | **0.5732** | **0.4018** | 84.0% | 62.5% | 1.47 | 0.91 |
| **screw** | Metal Hardware | 160 | **0.8776** | **0.4175** | **0.5885** | **0.4884** | **0.3231** | 84.9% | 65.9% | 2.28 | 2.00 |
| **transistor** | Electronics | 100 | **0.8200** | **0.4354** | **0.3898** | **0.4113** | **0.2589** | 62.5% | 76.7% | 2.69 | 0.88 |

*Detailed reports and defect distributions are available in [docs/phase3_initial_implementation.md](docs/phase3_initial_implementation.md) and [results/phase3/eda_report.md](results/phase3/eda_report.md).*

---

## Quick Start

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git
cd VisionInspect

# Create virtual environment & install requirements
python -m venv visioninspect_py312
.\visioninspect_py312\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Multi-Category Training

Train a PatchCore memory bank on normal images for any supported category:

```powershell
# Train on leather (automatically disables spatial prior for textures)
python scripts/train_patchcore.py --category leather

# Train on screw with custom coreset ratio
python scripts/train_patchcore.py --category screw --coreset-ratio 0.10
```

### 3. Multi-Category Evaluation & Threshold Calibration

Run 2D threshold sweep and metric evaluation:

```powershell
# Evaluate single category
python scripts/run_multi_category_eval.py --category leather

# Or evaluate all 5 categories
python scripts/run_multi_category_eval.py --category all
```

### 4. Run the Web Inspection Stack

Start the FastAPI backend:
```powershell
uvicorn app.backend:app --host 127.0.0.1 --port 8000
```

Start the Streamlit inspection UI:
```powershell
streamlit run app/app.py --server.port 8501
```

Access the dashboard at `http://localhost:8501`.

---

## API Reference

### `GET /health`
Returns system status, active PyTorch compute device, and list of cached models.

### `GET /categories`
Returns metadata, calibrated thresholds, and training status for all supported categories.

### `POST /predict`
Performs defect detection on an uploaded image.
- **Form Data**:
  - `file`: Product image (PNG, JPEG)
  - `category`: Category name (`bottle`, `leather`, `transistor`, `zipper`, `screw`)
  - `return_visualizations`: `true` or `false`
- **Response**:
  ```json
  {
    "category": "bottle",
    "is_defective": true,
    "status": "DEFECTIVE",
    "anomaly_score": 3.4812,
    "image_threshold": 1.50,
    "pixel_threshold": 1.40,
    "num_defects": 1,
    "defect_regions": [
      {
        "id": 1,
        "bbox": [210, 480, 85, 92],
        "area": 7820,
        "score": 3.1042,
        "aspect_ratio": 0.92
      }
    ],
    "visualization_base64": "...",
    "inference_time_ms": 78.4,
    "explanation": "Image classified as DEFECTIVE..."
  }
  ```

---

## Testing

VisionInspect includes a test suite covering models, detection logic, multi-category routing, and API endpoints:

```powershell
python -m pytest tests/ -v
```

Output:
```
======================= 35 passed in 24.37s =======================
```

---

## License

This project is licensed under the MIT License.
