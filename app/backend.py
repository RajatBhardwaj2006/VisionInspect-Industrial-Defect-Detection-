import os
import sys
import io
import time
import base64
import uuid
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configure secure server-side logging (stack traces stay in logs, never in client responses)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("visioninspect.backend")

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
from src.detection.category_checker import CategoryCompatibilityChecker
from src.utils.config import load_config, get_category_config

# Security Limits
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024       # 10 MB maximum upload
MAX_IMAGE_DIMENSION = 4096                      # 4096 px width/height limit against decompression bombs
MAX_IMAGE_PIXELS = 16_000_000                   # 16 megapixels max
RATE_LIMIT_WINDOW_SEC = 60                      # 1-minute window
RATE_LIMIT_INFERENCE_MAX = 60                   # Max 60 inferences/min per IP
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAGIC_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"RIFF": ".webp",
    b"BM": ".bmp",
}

# Pillow decompression bomb protection limit
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# Simple in-memory rate limiter per IP address
class InMemoryRateLimiter:
    def __init__(self, limit: int = 60, window_sec: int = 60):
        self.limit = limit
        self.window_sec = window_sec
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        req_times = self.requests[client_ip]
        # Discard records outside the sliding window
        self.requests[client_ip] = [t for t in req_times if now - t < self.window_sec]
        if len(self.requests[client_ip]) >= self.limit:
            return False
        self.requests[client_ip].append(now)
        return True

rate_limiter = InMemoryRateLimiter(limit=RATE_LIMIT_INFERENCE_MAX, window_sec=RATE_LIMIT_WINDOW_SEC)

# Security Headers & Throttling Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Apply rate limiting to expensive inference routes
        if request.url.path in ("/predict", "/predict/batch"):
            client_ip = request.client.host if request.client else "unknown"
            # Support X-Forwarded-For if behind a trusted reverse proxy
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            if not rate_limiter.is_allowed(client_ip):
                logger.warning(f"Rate limit exceeded for IP {client_ip} on {request.url.path}")
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Maximum 60 requests per minute allowed."}
                )

        response: Response = await call_next(request)
        
        # Hardened HTTP Security Headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "font-src 'self' data:; "
            "connect-src 'self' http: https: ws: wss:;"
        )
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
        return response

app = FastAPI(
    title="VisionInspect Industrial Defect Detection API",
    description="Multi-category industrial visual inspection powered by PatchCore v2.3",
    version="3.2.0",
    docs_url=None if os.environ.get("VISIONINSPECT_ENV") == "production" else "/docs",
    redoc_url=None if os.environ.get("VISIONINSPECT_ENV") == "production" else "/redoc",
)

# Attach Security Headers & Rate Limiting Middleware
app.add_middleware(SecurityHeadersMiddleware)

# Strict Environment-Configured CORS
configured_origins = os.environ.get("VISIONINSPECT_CORS_ORIGINS", "")
if configured_origins:
    origins = [orig.strip() for orig in configured_origins.split(",") if orig.strip()]
else:
    # Default local dev / demo origins
    origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

SUPPORTED_CATEGORIES = ["bottle", "leather", "transistor", "zipper", "screw"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In-memory detector cache to avoid reloading models on every request
DETECTOR_CACHE: Dict[str, PatchCoreDetectorV23] = {}
COMPATIBILITY_CHECKER: Optional[CategoryCompatibilityChecker] = None

# Global Exception Handler: Protect against internal error and stack trace leakage
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal inspection error occurred. Detailed diagnostic has been recorded in server logs."}
    )

def get_detector(category: str) -> PatchCoreDetectorV23:
    category = category.strip().lower()
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported category '{category}'. Initial supported categories: {SUPPORTED_CATEGORIES}"
        )
        
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    # Strictly map category to internal models directory (prevents path manipulation)
    model_dir = Path("models") / category / "patchcore_v23"
    if not model_dir.exists():
        logger.error(f"Model directory for category '{category}' not found at: {model_dir}")
        raise HTTPException(
            status_code=404,
            detail=f"Inspection model for category '{category}' is currently unavailable."
        )
        
    if category not in DETECTOR_CACHE:
        # Single-model active cache: Purge previous models to maintain lowest memory footprint
        if DETECTOR_CACHE:
            logger.info("Purging previously active model from memory to conserve system RAM...")
            DETECTOR_CACHE.clear()
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"Loading model for category '{category}' onto {DEVICE}...")
        DETECTOR_CACHE[category] = PatchCoreDetectorV23(category=category, device=DEVICE)
        
    return DETECTOR_CACHE[category]

def get_compatibility_checker() -> CategoryCompatibilityChecker:
    global COMPATIBILITY_CHECKER
    if COMPATIBILITY_CHECKER is None:
        # Reuse existing feature extractor if an active detector is already loaded
        shared_fe = None
        if DETECTOR_CACHE:
            active_det = next(iter(DETECTOR_CACHE.values()))
            shared_fe = getattr(active_det.model, "feature_extractor", None)
        COMPATIBILITY_CHECKER = CategoryCompatibilityChecker(device=DEVICE, feature_extractor=shared_fe)
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
        model_dir = Path("models") / cat / "patchcore_v23"
        cat_details.append({
            "category": cat,
            "status": "trained" if model_dir.exists() else "untrained",
            "image_threshold": cat_cfg.get("image_threshold", 1.50),
            "pixel_threshold": cat_cfg.get("pixel_threshold", 1.40),
            "use_spatial_prior": cat_cfg.get("use_spatial_prior", True),
            "backbone": "resnet18",
            "model_dir": f"models/{cat}/patchcore_v23/"
        })
    return {"categories": cat_details}

def _validate_image_file(file_bytes: bytes, filename: str) -> Image.Image:
    """
    Validates uploaded file against size, MIME magic bytes, decompression bombs,
    and decoding errors.
    """
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"Uploaded file size ({len(file_bytes) / (1024*1024):.1f} MB) exceeds maximum allowed limit of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB."
        )

    # Sanitize and validate filename extension
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats: {sorted(ALLOWED_EXTENSIONS)}")

    # Magic byte header verification
    is_valid_magic = False
    for magic in MAGIC_SIGNATURES:
        if file_bytes.startswith(magic):
            is_valid_magic = True
            break
    if not is_valid_magic:
        raise ValueError("Invalid image file: File content does not match a valid image signature.")

    # Image decoding and dimension verification (Decompression bomb protection)
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except Exception as e:
        raise ValueError("Invalid image file: Corrupted or unreadable image data.")

    try:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError("Invalid image file: Failed to decode image data into RGB.")

    w, h = pil_image.size
    if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions ({w}x{h}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} px."
        )
    if (w * h) > MAX_IMAGE_PIXELS:
        raise ValueError(f"Total pixel count ({w * h}) exceeds maximum limit.")

    return pil_image

def _process_single_image(
    file_bytes: bytes,
    filename: str,
    category: str,
    detector: PatchCoreDetectorV23,
    checker: CategoryCompatibilityChecker,
    return_visualizations: bool = True
) -> Dict[str, Any]:
    """Internal helper to process a single image with defensive error handling."""
    # Sanitize user-provided filename against path traversal
    safe_filename = Path(filename).name if filename else "image.png"

    # Strict multi-layer validation
    pil_image = _validate_image_file(file_bytes, safe_filename)

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
            "label": reg.get("label", f"Region {idx+1:02d}"),
            "intensity": reg.get("intensity", "Anomaly Region"),
            "bbox": [reg["x"], reg["y"], w, h],
            "area": reg["area"],
            "score": round(float(reg.get("score", 0.0)), 4),
            "aspect_ratio": round(float(w / max(1, h)), 2)
        })

    req_id = f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    vis_base64 = result.get("visualization_base64")
    if not vis_base64 and return_visualizations and "saved_path" in result:
        saved_p = Path(result["saved_path"])
        if saved_p.exists():
            with open(saved_p, "rb") as f:
                vis_base64 = base64.b64encode(f.read()).decode("utf-8")

    heatmap_base64 = result.get("heatmap_base64")
    decision_margin = result.get("decision_margin", round(float(result["score"] - result["image_threshold"]), 4))

    # Calculate peak anomaly score across localized regions or raw score
    peak_score = round(float(max([reg.get("score", 0.0) for reg in defect_regions] + [result.get("score", 0.0)])), 4)

    # Extract memory bank size if available
    mem_size = "N/A"
    if hasattr(detector, "model") and hasattr(detector.model, "memory_bank"):
        mb = detector.model.memory_bank
        if mb is not None:
            mem_size = f"{len(mb):,} patch vectors (10% coreset)"

    return {
        "request_id": req_id,
        "filename": safe_filename,
        "category": category,
        "model": "PatchCore v2.3",
        "model_path": f"models/{category}/patchcore_v23/",
        "status": result["status"],
        "is_defective": result["status"] == "DEFECTIVE",
        "anomaly_score": round(float(result["score"]), 4),
        "image_threshold": round(float(result["image_threshold"]), 4),
        "pixel_threshold": round(float(result["pixel_threshold"]), 4),
        "decision_margin": decision_margin,
        "peak_anomaly": peak_score,
        "memory_bank": mem_size,
        "num_defects": len(defect_regions),
        "localized_regions": defect_regions,
        "defect_regions": defect_regions,
        "heatmap_base64": heatmap_base64,
        "visualization_base64": vis_base64,
        "inference_time_ms": round(t_elapsed, 2),
        "inference_time_s": round(t_elapsed / 1000.0, 2),
        "explanation": result["explanation"],
        "why_explanation": result.get("why_explanation", ""),
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
        logger.error(f"Error during inspection for category '{category}': {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred during defect inspection.")

    return res

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    category: str = Form("bottle"),
    return_visualizations: bool = Form(True)
):
    """
    Batch inspection endpoint for 1 to 5 images.
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
            logger.error(f"Batch image failure for {fname}: {str(e)}", exc_info=True)
            batch_results.append({
                "filename": fname,
                "status": "ERROR",
                "is_defective": False,
                "error": "Inspection failed for this image."
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
