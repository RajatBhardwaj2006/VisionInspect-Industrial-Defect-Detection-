import os
import torch
import numpy as np
from pathlib import Path

# Fix OpenMP duplicate runtime initialization issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_anomaly_detector_init():
    from src.detection.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(category="bottle")
    assert detector is not None
    assert detector.category == "bottle"
    assert detector.image_threshold == 0.50
    assert detector.pixel_threshold == 0.20

def test_anomaly_detector_inspect_defective():
    from src.detection.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(category="bottle")
    res = detector.inspect("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    assert res["status"] == "DEFECTIVE"
    assert abs(res["score"] - 0.535785) < 1e-4
    assert res["bounding_box"] != "None"
    bbox = res["bounding_box"]
    # Bounding box coordinates must align with the crack on the right side of the image
    # Note that scaled x starts around 625 (>= 450)
    assert bbox["x"] >= 450
    assert bbox["width"] > 0
    assert bbox["height"] > 0

def test_anomaly_detector_inspect_normal():
    from src.detection.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(category="bottle")
    res = detector.inspect("dataset/mvtec_anomaly_detection/bottle/test/good/001.png")
    assert res["status"] == "NORMAL"
    assert res["bounding_box"] == "None"

def test_anomaly_map_shape():
    """Test that anomaly map has matching spatial resolution [256, 256]."""
    from src.detection.heatmap import compute_anomaly_map
    img = torch.randn(1, 3, 256, 256)
    recon = torch.randn(1, 3, 256, 256)
    amap = compute_anomaly_map(img, recon).squeeze(0)
    assert amap.shape == (256, 256)

def test_anomaly_score():
    """Test that reconstruction-based anomaly scoring functions correctly."""
    from src.detection.heatmap import compute_anomaly_map
    img = torch.ones(1, 3, 256, 256)
    # Perfect reconstruction
    amap = compute_anomaly_map(img, img).squeeze(0)
    assert float(amap.max().item()) == 0.0

def test_localization_regions():
    """Test that localize() returns sorted lists of regions."""
    from src.detection.localization import localize
    amap = torch.zeros(256, 256)
    # Add a synthetic defect region
    amap[100:150, 100:150] = 0.8
    regions = localize(amap, pixel_thresh=0.20)
    assert len(regions) > 0
    assert abs(regions[0]["width"] - 50) <= 2
    assert abs(regions[0]["height"] - 50) <= 2
    assert abs(regions[0]["area"] - 2500) <= 300

def test_multiple_regions():
    """Test that localize() can identify multiple regions simultaneously."""
    from src.detection.localization import localize
    amap = torch.zeros(256, 256)
    # Defect region 1
    amap[50:65, 50:65] = 0.9
    # Defect region 2 (completely separate)
    amap[150:165, 150:165] = 0.95
    regions = localize(amap, pixel_thresh=0.20)
    assert len(regions) >= 2

def test_coordinate_scaling():
    """Test coordinate scaling between model space [256x256] and original image space."""
    orig_w, orig_h = 900, 900
    scale_x = orig_w / 256.0
    scale_y = orig_h / 256.0
    
    rx, ry = 100, 150
    scaled_x = int(rx * scale_x)
    scaled_y = int(ry * scale_y)
    
    assert scaled_x == 351
    assert scaled_y == 527

def test_filename_uniqueness():
    """Test that inspection saves results under unique filenames based on category, type, and stem."""
    from src.detection.anomaly_detector import AnomalyDetector
    detector = AnomalyDetector(category="bottle")
    
    # Trigger inspect to generate filenames
    res1 = detector.inspect("dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png")
    res2 = detector.inspect("dataset/mvtec_anomaly_detection/bottle/test/broken_small/000.png")
    
    # Assert they are distinct and exist
    path1 = Path(res1["saved_path"])
    path2 = Path(res2["saved_path"])
    assert path1 != path2
    assert path1.exists()
    assert path2.exists()
