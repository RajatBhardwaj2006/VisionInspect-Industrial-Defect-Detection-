import os
import sys
import io
import time
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
from src.detection.category_checker import CategoryCompatibilityChecker
from src.utils.config import load_config, get_category_config

app = FastAPI(
    title="VisionInspect Industrial Defect Detection API",
    description="Multi-category industrial visual inspection powered by PatchCore v2.3",
    version="3.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_CATEGORIES = ["bottle", "leather", "transistor", "zipper", "screw"]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In-memory detector cache to avoid reloading models on every request
DETECTOR_CACHE: Dict[str, PatchCoreDetectorV23] = {}
COMPATIBILITY_CHECKER: Optional[CategoryCompatibilityChecker] = None

def get_detector(category: str) -> PatchCoreDetectorV23:
    category = category.strip().lower()
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported category '{category}'. Initial supported categories: {SUPPORTED_CATEGORIES}"
        )
        
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))
    if not model_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model directory for category '{category}' not found at: {model_dir}. Please train the model first."
        )
        
    if category not in DETECTOR_CACHE:
        print(f"[Backend] Loading model for category '{category}' onto {DEVICE}...")
        DETECTOR_CACHE[category] = PatchCoreDetectorV23(category=category, device=DEVICE)
        
    return DETECTOR_CACHE[category]

def get_compatibility_checker() -> CategoryCompatibilityChecker:
    global COMPATIBILITY_CHECKER
    if COMPATIBILITY_CHECKER is None:
        COMPATIBILITY_CHECKER = CategoryCompatibilityChecker(device=DEVICE)
    return COMPATIBILITY_CHECKER

@app.get("/")
def root():
    """Root entrypoint providing service status and API documentation links."""
    return {
        "service": "VisionInspect Industrial Defect Detection API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "categories": "/categories",
        "version": "3.2.0"
    }

@app.get("/health")
def health_check():
    """Health check endpoint exposing system status, device, and loaded models."""
    return {
        "status": "ok",
        "supported_categories": SUPPORTED_CATEGORIES,
        "loaded_models": list(DETECTOR_CACHE.keys()),
        "device": str(DEVICE),
        "version": "3.2.0"
    }

@app.get("/categories")
def get_categories():
    """Returns supported categories and their configuration parameters."""
    cfg = load_config()
    cat_details = []
    for cat in SUPPORTED_CATEGORIES:
        cat_cfg = get_category_config(cat, cfg)
        model_dir = Path(cat_cfg.get("model_dir", f"models/{cat}/patchcore_v23"))
        cat_details.append({
            "category": cat,
            "status": "trained" if model_dir.exists() else "untrained",
            "image_threshold": cat_cfg.get("image_threshold", 1.50),
            "pixel_threshold": cat_cfg.get("pixel_threshold", 1.40),
            "use_spatial_prior": cat_cfg.get("use_spatial_prior", True),
            "backbone": "resnet18",
            "model_dir": f"models/{cat}/patchcore_v23"
        })
    return {"categories": cat_details}

def _process_single_image(
    file_bytes: bytes,
    filename: str,
    category: str,
    detector: PatchCoreDetectorV23,
    checker: CategoryCompatibilityChecker,
    return_visualizations: bool = True
) -> Dict[str, Any]:
    """Internal helper to process a single image with error handling."""
    # Check filename extension if provided
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: {sorted(ALLOWED_EXTENSIONS)}")
        
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")
        
    try:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")
        
    t_start = time.perf_counter()
    result = detector.inspect(pil_image)
    t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
    
    # Run category compatibility check
    comp_result = checker.check_compatibility(pil_image, category)
    
    # Build region payload
    defect_regions = []
    for idx, reg in enumerate(result.get("localized_regions", [])):
        w = reg["width"]
        h = reg["height"]
        defect_regions.append({
            "id": idx + 1,
            "bbox": [reg["x"], reg["y"], w, h],
            "area": reg["area"],
            "score": round(float(reg.get("score", 0.0)), 4),
            "aspect_ratio": round(float(w / max(1, h)), 2)
        })
        
    vis_base64 = None
    if return_visualizations and "saved_path" in result:
        saved_p = Path(result["saved_path"])
        if saved_p.exists():
            with open(saved_p, "rb") as f:
                vis_base64 = base64.b64encode(f.read()).decode("utf-8")
                
    return {
        "filename": Path(filename).name if filename else "image.png",
        "category": category,
        "model": "PatchCore v2.3",
        "model_path": f"models/{category}/patchcore_v23/",
        "status": result["status"],
        "is_defective": result["status"] == "DEFECTIVE",
        "anomaly_score": round(float(result["score"]), 4),
        "image_threshold": round(float(result["image_threshold"]), 4),
        "pixel_threshold": round(float(result["pixel_threshold"]), 4),
        "num_defects": len(defect_regions),
        "localized_regions": defect_regions,
        "defect_regions": defect_regions,
        "visualization_base64": vis_base64,
        "inference_time_ms": round(t_elapsed, 2),
        "explanation": result["explanation"],
        "category_compatibility": comp_result,
        "compatibility": comp_result,
        "category_warning": comp_result.get("warning_message")
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    category: str = Form("bottle"),
    return_visualizations: bool = Form(True)
):
    """
    Run industrial defect inspection on a single uploaded image.
    Includes category compatibility check to warn if image appears more compatible with another product.
    """
    category = category.strip().lower()
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported category '{category}'. Supported categories: {SUPPORTED_CATEGORIES}"
        )
    detector = get_detector(category)
    checker = get_compatibility_checker()
    
    content = await file.read()
    try:
        res = _process_single_image(
            file_bytes=content,
            filename=file.filename or "image.png",
            category=category,
            detector=detector,
            checker=checker,
            return_visualizations=return_visualizations
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
        
    return res

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    category: str = Form("bottle"),
    return_visualizations: bool = Form(True)
):
    """
    Batch inspection endpoint for 1 to 5 images.
    Rejects batches larger than 5 images.
    Processes images independently with per-image error resilience.
    """
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch size must be between 1 and 5 images. Upload at least one image to begin inspection."
        )
        
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size must be between 1 and 5 images. Please upload a maximum of 5 images. You uploaded {len(files)} images."
        )
        
    category = category.strip().lower()
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported category '{category}'. Supported categories: {SUPPORTED_CATEGORIES}"
        )

    detector = get_detector(category)
    checker = get_compatibility_checker()
    
    batch_results = []
    for f in files:
        fname = Path(f.filename or "image.png").name
        try:
            content = await f.read()
            img_result = _process_single_image(
                file_bytes=content,
                filename=fname,
                category=category,
                detector=detector,
                checker=checker,
                return_visualizations=return_visualizations
            )
            batch_results.append(img_result)
        except ValueError as ve:
            batch_results.append({
                "filename": fname,
                "status": "ERROR",
                "is_defective": False,
                "error": str(ve)
            })
        except Exception as e:
            batch_results.append({
                "filename": fname,
                "status": "ERROR",
                "is_defective": False,
                "error": f"Inspection failed: {str(e)}"
            })
            
    # Calculate summary counts
    normal_count = sum(1 for r in batch_results if r.get("status") == "NORMAL")
    defective_count = sum(1 for r in batch_results if r.get("status") == "DEFECTIVE")
    warning_count = sum(
        1 for r in batch_results
        if r.get("category_warning") or (r.get("category_compatibility") and r["category_compatibility"].get("is_mismatch"))
    )
    failed_count = sum(1 for r in batch_results if r.get("status") == "ERROR")

    summary = {
        "total": len(files),
        "normal": normal_count,
        "defective": defective_count,
        "category_warnings": warning_count,
        "failed": failed_count
    }

    return {
        "category": category,
        "model": "PatchCore v2.3",
        "model_path": f"models/{category}/patchcore_v23/",
        "batch_size": len(files),
        "total_images": len(files),
        "summary": summary,
        "results": batch_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend:app", host="127.0.0.1", port=8000, reload=False)
