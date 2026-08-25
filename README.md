# VisionInspect

### Industrial Defect Detection and Localization using Unsupervised Learning

VisionInspect is an AI-powered industrial visual inspection system designed to detect and localize product defects using **unsupervised anomaly detection**.

The system learns the visual appearance of defect-free products and identifies unusual regions in previously unseen images. When an anomaly is detected, VisionInspect provides an **anomaly score, visual heatmap, defect location, and inspection explanation**.

The project is based on the **MVTec AD Industrial Anomaly Detection Dataset** and is developed as a project-based assignment for Unsupervised Learning.

---

## Project Overview

In industrial manufacturing, manually inspecting every product for scratches, cracks, dents, contamination, and other defects can be expensive and time-consuming.

A major challenge is that:

* Defective products are relatively rare.
* It is difficult to collect examples of every possible defect.
* New and previously unseen defect types may appear.
* Manually labeling large image datasets is expensive.

VisionInspect addresses this problem by learning primarily from **normal/defect-free images** and detecting images or regions that differ significantly from the learned normal appearance.

---

## Key Features

### Single Image Inspection

Upload an individual product image and receive:

* Normal / Defective prediction
* Anomaly score
* Anomaly heatmap
* Defect location
* Highlighted defect region
* Short inspection explanation

### Batch Inspection

Upload multiple images at once and process them together.

The system provides:

* Total images inspected
* Number of normal images
* Number of defective images
* Individual anomaly scores
* Individual predictions
* Confidence/results summary

### Visual Anomaly Localization

VisionInspect generates an anomaly heatmap showing the regions that contribute most strongly to the anomaly score.

The interface can display:

```text
Original Image
       +
Anomaly Heatmap
       +
Defect Highlight
```

### Evaluation

The system evaluates anomaly-detection performance using appropriate metrics such as:

* Image-level AUROC
* Precision
* Recall
* F1-score
* Confusion Matrix
* Pixel-level localization metrics where applicable

---

## Machine Learning Approach

The initial implementation uses a **Convolutional Autoencoder** for unsupervised anomaly detection.

### Basic Pipeline

```text
MVTec AD Dataset
       ↓
Image Preprocessing
       ↓
Normal Training Images
       ↓
Convolutional Autoencoder
       ↓
Learn Normal Product Appearance
       ↓
Test Image
       ↓
Image Reconstruction
       ↓
Reconstruction Error
       ↓
Anomaly Score
       ↓
Normal / Defective
       ↓
Anomaly Map
       ↓
Defect Localization
```

The model is trained primarily using defect-free images.

During inference, the model attempts to reconstruct the input image. If an image contains an unusual region that the model cannot reconstruct accurately, the reconstruction error in that region becomes higher.

---

## Dataset

VisionInspect uses the **MVTec AD — Industrial Anomaly Detection Dataset**.

The dataset contains industrial images belonging to multiple product categories, including:

* Bottle
* Cable
* Capsule
* Carpet
* Grid
* Hazelnut
* Leather
* Metal Nut
* Pill
* Screw
* Tile
* Toothbrush
* Transistor
* Wood
* Zipper

The dataset contains normal training images and normal/defective test images together with anomaly annotations.

Official dataset:

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

> The complete dataset should not be committed to this repository because of its large size.

---

## Example Output

The final VisionInspect interface is designed to provide an inspection result similar to:

```text
┌──────────────────────────────────────────────┐
│              INSPECTION RESULT               │
├─────────────────────┬────────────────────────┤
│                     │                        │
│   Original Image    │    Anomaly Heatmap     │
│                     │                        │
│       Product       │       Heatmap          │
│        🔴           │        🔴🔴             │
│                     │                        │
├─────────────────────┴────────────────────────┤
│                                             │
│  Status:          DEFECTIVE                 │
│  Anomaly Score:   0.89                      │
│  Confidence:      89%                       │
│                                             │
│  Defect Location: Detected region           │
│                                             │
│  Explanation:                               │
│  An abnormal region was detected on the     │
│  surface of the inspected product.          │
│                                             │
└─────────────────────────────────────────────┘
```

The exact values shown above are examples only.

---

## Project Structure

```text
VisionInspect/
│
├── dataset/
│   └── mvtec_anomaly_detection/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_autoencoder_training.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── detection/
│   ├── evaluation/
│   └── utils/
│
├── models/
│   └── trained_models/
│
├── results/
│   ├── predictions/
│   ├── heatmaps/
│   ├── comparisons/
│   └── metrics/
│
├── app/
│   ├── app.py
│   ├── components/
│   └── styles/
│
├── tests/
│
├── docs/
│
├── requirements.txt
├── config.yaml
├── .gitignore
└── README.md
```

---

## Technology Stack

### Machine Learning

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn

### Computer Vision

* OpenCV
* Pillow
* Matplotlib

### Visualization

* Matplotlib
* Plotly

### Application

The final interface is intended to run as a local web application initially, with deployment considered after the model and application are stable.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd VisionInspect
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the MVTec AD dataset inside:

```text
dataset/mvtec_anomaly_detection/
```

---

## Running the Project

The development workflow is divided into several stages.

### 1. Explore the Dataset

```bash
python <dataset-exploration-script>
```

This stage checks the dataset structure and displays normal and defective samples.

### 2. Train the Model

```bash
python <training-script>
```

The trained model will be saved inside the models directory.

### 3. Run Anomaly Detection

```bash
python <detection-script>
```

The system calculates anomaly scores and generates anomaly maps.

### 4. Start VisionInspect

```bash
streamlit run app/app.py
```

The application will open in the browser and provide the VisionInspect inspection interface.

---

## Inspection Workflow

A typical inspection follows:

```text
Upload Image
     ↓
Preprocessing
     ↓
Model Inference
     ↓
Image Reconstruction
     ↓
Reconstruction Error
     ↓
Anomaly Score
     ↓
Threshold Comparison
     ↓
Normal / Defective
     ↓
Anomaly Localization
     ↓
Heatmap + Defect Region
     ↓
Inspection Report
```

For batch inspection:

```text
Upload Multiple Images
          ↓
    Process Images
          ↓
┌─────────┼─────────┐
↓         ↓         ↓
Image 1  Image 2  Image N
↓         ↓         ↓
Result   Result   Result
└─────────┼─────────┘
          ↓
   Batch Summary
```

---

## Important Design Principle

VisionInspect is designed as an **anomaly-detection system**, not simply a conventional image classifier.

The primary goal is to learn:

> **What does a normal product look like?**

and then determine:

> **How different is this new product from the learned normal appearance?**

This allows the system to identify unusual patterns without requiring a large collection of labeled examples for every possible defect.

---

## Current Development Plan

The project will be developed incrementally:

```text
Phase 1
Dataset Exploration
        ↓
Phase 2
Image Preprocessing
        ↓
Phase 3
Convolutional Autoencoder
        ↓
Phase 4
Model Training
        ↓
Phase 5
Anomaly Score
        ↓
Phase 6
Anomaly Heatmap
        ↓
Phase 7
Defect Localization
        ↓
Phase 8
Model Evaluation
        ↓
Phase 9
VisionInspect Web Interface
        ↓
Phase 10
Batch Inspection
        ↓
Phase 11
Reports & Visualization
        ↓
Phase 12
Deployment
```

---

## Future Improvements

Possible future improvements include:

* PatchCore-style feature-based anomaly detection
* Improved defect localization
* Better threshold selection
* Multiple product categories
* Batch inspection
* Automated inspection reports
* Model comparison
* GPU acceleration
* Cloud deployment
* REST API
* Production monitoring

---

## Disclaimer

VisionInspect is an academic/project-based prototype intended to demonstrate unsupervised industrial anomaly detection. It should not be used as the sole decision-making system for real industrial quality-control operations without appropriate validation and safety testing.

---

## Project Goal

The ultimate goal of VisionInspect is to create a practical visual inspection system that can:

**See → Detect → Localize → Explain → Report**

anomalies in industrial products using unsupervised learning.
