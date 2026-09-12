import os
import sys
import argparse
from pathlib import Path

# Fix OpenMP duplicate runtime initialization issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.anomaly_detector import AnomalyDetector
from src.detection.patchcore_detector import PatchCoreDetector
from src.detection.patchcore_v22_detector import PatchCoreDetectorV22
from src.detection.patchcore_v23_detector import PatchCoreDetectorV23

def main():
    parser = argparse.ArgumentParser(description="VisionInspect - Anomaly Detection Inference")
    parser.add_argument("--category", type=str, required=True, help="Object category (e.g. bottle)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, choices=["autoencoder", "patchcore", "patchcore_v22", "patchcore_v23"], default="patchcore_v23", help="Detector model architecture")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Input image file not found at {args.image}", file=sys.stderr)
        sys.exit(1)
        
    try:
        if args.model == "patchcore_v23":
            detector = PatchCoreDetectorV23(category=args.category)
        elif args.model == "patchcore_v22":
            detector = PatchCoreDetectorV22(category=args.category)
        elif args.model == "patchcore":
            detector = PatchCoreDetector(category=args.category)
        else:
            detector = AnomalyDetector(category=args.category)
    except Exception as e:
        print(f"Error initializing detector ({args.model}): {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Run inspection
        results = detector.inspect(str(image_path))
    except Exception as e:
        print(f"Error during inspection: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Output results to terminal as required
    print("\n========================================")
    print("        VISIONINSPECT INFERENCE         ")
    print("========================================")
    print(f"Requested Category:   {args.category}")
    print(f"Loaded Category:      {results.get('loaded_category', args.category)}")
    print(f"Model Architecture:   {results.get('model_version', args.model)}")
    print(f"Memory Bank Path:     {results.get('model_dir', 'N/A')}")
    print(f"Spatial Prior Used:   {results.get('use_spatial_prior', 'N/A')}")
    print(f"Image:                {args.image}")
    print(f"Result Status:        {results['status']}")
    print(f"Anomaly Score:        {results['score']:.6f}")
    print(f"Image Threshold:      {results['image_threshold']:.6f}")
    print(f"Pixel Threshold:      {results['pixel_threshold']:.6f}")
    
    # Report Defect Localization
    bbox = results['bounding_box']
    if bbox and bbox != "None":
        print(f"Defect Localized: YES")
        print(f"  Bounding Box:   x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, height={bbox['height']}")
        print(f"  Centroid:       ({bbox['centroid'][0]:.2f}, {bbox['centroid'][1]:.2f})")
    else:
        print("Defect Localized: NO")
        
    print(f"Explanation:      {results['explanation']}")
    
    # Heatmap & Highlighted Image generation verification
    saved_path = Path(results['saved_path'])
    if saved_path.exists():
        print(f"Heatmap & Highlighted Image Saved: YES")
        print(f"  Saved Path:     {saved_path.resolve()}")
    else:
        print("Heatmap & Highlighted Image Saved: NO")
    print("========================================\n")

if __name__ == "__main__":
    main()