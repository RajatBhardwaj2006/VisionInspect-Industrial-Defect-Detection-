# VisionInspect

### Industrial Defect Detection and Localization using Unsupervised Learning

VisionInspect is an AI-powered industrial visual inspection system designed to detect and localize product defects using **unsupervised anomaly detection**.

The system learns the visual appearance of defect-free products and identifies unusual regions in previously unseen images. When an anomaly is detected, VisionInspect provides an **anomaly score, visual heatmap, defect location, and inspection explanation**.

The project is based on the **MVTec AD Industrial Anomaly Detection Dataset** and is developed as a project-based assignment for Unsupervised Learning.

---

## Application Preview

VisionInspect is designed as a modern industrial inspection dashboard with single-image inspection, batch inspection, anomaly visualization, and performance reporting.

The inspection screen provides:

* Original image
* Anomaly heatmap
* Defect highlighted on the original image
* Anomaly score
* Confidence
* Detection threshold
* Defect status
* Defect location
* Short defect explanation

### Complete Application Overview

The application is designed to provide separate interfaces for inspection, batch inspection, and performance analysis.
### Inspection Result

![VisionInspect Inspection Result](Docs/images/inspection-result.png)

### Complete Application Overview

![VisionInspect Application Overview](Docs/images/application-overview.png)
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
* Results summary

### Visual Anomaly Localization

VisionInspect generates an anomaly heatmap showing the regions that contribute most strongly to the anomaly score.

The interface displays:

```text
Original Image
       +
Anomaly Heatmap
       +
Defect Highlight
```

### Performance Reports

The application is designed to provide model-performance information including:

* Image-level AUROC
* Pixel-level localization metrics where applicable
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC Curve
* Anomaly Score Distribution

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
       ↓
Inspection Result
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

### Batch Inspection Workflow

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

## Example Inspection Result

A defective product can be presented to the user with:

```text
Status:          DEFECTIVE

Anomaly Score:   0.89

Confidence:      89%

Threshold:       0.50

Defect Location:
X: 612 - 760
Y: 480 - 880

Explanation:
An abnormal region was detected on the
surface of the inspected product.
```

The values above are example interface values and are not fixed model results.

---

## Project Architecture

```text
                    ┌─────────────────────┐
                    │     User Image      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Convolutional       │
                    │ Autoencoder         │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Image Reconstruction│
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Reconstruction      │
                    │ Error               │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Anomaly Score       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Normal / Defective  │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Anomaly Heatmap     │
                    │ & Localization      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ VisionInspect UI    │
                    └─────────────────────┘
```

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
│   │   ├── dataset_loader.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── autoencoder.py
│   │   └── feature_extractor.py
│   │
│   ├── detection/
│   │   ├── anomaly_detector.py
│   │   ├── anomaly_score.py
│   │   ├── heatmap.py
│   │   └── localization.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── roc_curve.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── config.py
│       └── helpers.py
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
│   │   ├── upload.py
│   │   ├── inspection.py
│   │   ├── heatmap_viewer.py
│   │   ├── batch_results.py
│   │   └── reports.py
│   │
│   └── styles/
│       └── style.css
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_detection.py
│   └── test_app.py
│
├── docs/
│   ├── images/
│   │   ├── inspection-result.png
│   │   └── application-overview.png
│   │
│   ├── architecture.png
│   └── workflow.png
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

The application is intended to run locally during development and can later be deployed as a web-based industrial inspection system.

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

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

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

### 1. Explore the Dataset

The first stage is to inspect the MVTec AD dataset and verify its structure.

```bash
python <dataset-exploration-script>
```

This stage displays normal and defective samples and verifies that the dataset is loaded correctly.

### 2. Train the Model

```bash
python <training-script>
```

The trained model will be saved inside the `models/` directory.

### 3. Run Anomaly Detection

```bash
python <detection-script>
```

The system calculates anomaly scores and generates anomaly maps.

### 4. Start VisionInspect

```bash
streamlit run app/app.py
```

The VisionInspect web application will open in the browser.

---

## Model Training Strategy

VisionInspect follows an unsupervised anomaly-detection approach.

The model learns primarily from **defect-free images**.

```text
Normal Images
     ↓
Convolutional Autoencoder
     ↓
Learn Normal Appearance
     ↓
New Image
     ↓
Reconstruction
     ↓
Compare Input vs Reconstruction
     ↓
Reconstruction Error
     ↓
Anomaly Score
```

A sufficiently different image or image region produces a larger reconstruction error and can therefore be identified as anomalous.

---

## Anomaly Localization

The anomaly map is generated from the difference between the input image and its reconstruction.

Conceptually:

```text
Original Image
      ↓
      ├───────────────┐
      ↓               ↓
Autoencoder     Reconstruction
      │               │
      └───────┬───────┘
              ↓
       Pixel Difference
              ↓
        Anomaly Map
              ↓
       Heatmap / Region
```

The highest-scoring regions are used to identify the location of the suspected anomaly.

---

## Evaluation

VisionInspect will evaluate the model using appropriate anomaly-detection metrics.

### Image-Level Evaluation

* AUROC
* Precision
* Recall
* F1-score
* Confusion Matrix

### Pixel-Level Evaluation

Where applicable:

* Pixel-level AUROC
* Anomaly localization metrics

The evaluation results will be stored in the `results/metrics/` directory.

---

## Important Design Principle

VisionInspect is designed as an **anomaly-detection system**, rather than simply a conventional image classifier.

The primary question is:

> **What does a normal product look like?**

The system then evaluates:

> **How different is this new product from the learned normal appearance?**

This approach allows the system to detect unusual patterns without requiring a large labeled dataset containing every possible defect.

---

## Development Roadmap

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
* Support for multiple product categories
* Batch inspection
* Automated inspection reports
* Model comparison
* GPU acceleration
* REST API
* Cloud deployment
* Production monitoring

---

## Deployment

The first version of VisionInspect will run locally.

```text
User
 ↓
VisionInspect Web App
 ↓
Trained Model
 ↓
Prediction
 ↓
Heatmap
 ↓
Inspection Result
```

After the model and application are stable, the system can be prepared for deployment.

The complete MVTec AD dataset does not need to be included in the deployed application. The trained model can be stored separately and used for inference.

---

## Limitations

VisionInspect is an academic/project-based prototype.

The system may require additional validation before being used in a real industrial production environment.

Potential limitations include:

* Dependence on image quality
* Sensitivity to lighting and background changes
* Threshold selection
* Reconstruction quality
* Generalization to completely different products
* Difficulty identifying semantic defect types using anomaly detection alone

---

## Project Goal

The ultimate goal of VisionInspect is to create a practical visual inspection system that can:

**See → Detect → Localize → Explain → Report**

industrial product anomalies using unsupervised learning.

---

## Disclaimer

VisionInspect is an academic/project-based prototype intended to demonstrate unsupervised industrial anomaly detection. It should not be used as the sole decision-making system for real industrial quality-control operations without appropriate validation and safety testing.

---

## License

This project is intended for educational and academic purposes.

The MVTec AD dataset is subject to its own licensing and usage conditions. Please refer to the official dataset documentation before redistributing or using the dataset.
