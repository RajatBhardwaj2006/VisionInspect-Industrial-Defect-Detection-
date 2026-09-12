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
from src.utils.config import load_config, get_category_config

app = FastAPI(
    title="VisionInspect Industrial Defect Detection API",
    description="Multi-category industrial visual inspection powered by PatchCore v2.3",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_CATEGORIES = ["bottle", "leather", "transistor", "zipper", "screw"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In-memory detector cache to avoid reloading models on every request
DETECTOR_CACHE: Dict[str, PatchCoreDetectorV23] = {}

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

@app.get("/health")
def health_check():
    """Health check endpoint exposing system status, device, and loaded models."""
    return {
        "status": "ok",
        "supported_categories": SUPPORTED_CATEGORIES,
        "loaded_models": list(DETECTOR_CACHE.keys()),
        "device": str(DEVICE),
        "version": "3.0.0"
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
            "model_dir": str(model_dir)
        })
    return {"categories": cat_details}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    category: str = Form("bottle"),
    return_visualizations: bool = Form(True)
):
    """
    Run industrial defect inspection on an uploaded image.
    Returns anomaly classification, calibrated threshold, defect count, bounding boxes, and visual overlay.
    """
    category = category.strip().lower()
    detector = get_detector(category)
    
    # Read & validate uploaded image
    try:
        content = await file.read()
        pil_image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
    t_start = time.perf_counter()
    try:
        result = detector.inspect(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    t_elapsed = (time.perf_counter() - t_start) * 1000.0  # ms
    
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
        "category": category,
        "is_defective": result["status"] == "DEFECTIVE",
        "status": result["status"],
        "anomaly_score": round(float(result["score"]), 4),
        "image_threshold": round(float(result["image_threshold"]), 4),
        "pixel_threshold": round(float(result["pixel_threshold"]), 4),
        "num_defects": len(defect_regions),
        "defect_regions": defect_regions,
        "visualization_base64": vis_base64,
        "inference_time_ms": round(t_elapsed, 2),
        "explanation": result["explanation"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend:app", host="127.0.0.1", port=8000, reload=False)
