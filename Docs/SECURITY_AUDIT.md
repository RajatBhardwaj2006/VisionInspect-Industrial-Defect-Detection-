# VisionInspect — Production Security Audit & Hardening Report

> **Document Type:** Production Security Audit & Verification  
> **Target Environment:** Public Cloud Deployment / Edge IPC / Docker Container  
> **Application Version:** 3.2.0  
> **Date:** October 2026  
> **Status:** **PASS** (Zero critical vulnerabilities; hardened defensive controls verified)

---

## 1. Executive Summary

Prior to public deployment, VisionInspect underwent an end-to-end security audit and defensive hardening pass covering all application tiers:
- **FastAPI REST API Service** (`app/backend.py`)
- **Streamlit Web User Interface** (`app/app.py`)
- **ReportLab PDF Generator** (`app/utils/pdf_report.py`)
- **PatchCore v2.3 Inference Pipeline** (`src/detection/patchcore_v23_detector.py`)
- **Environment & Secret Configuration** (`.env.example`, Git history)
- **Container Infrastructure** (`Dockerfile`, `.dockerignore`)

The hardening pass verified that no machine secrets, passwords, or credentials exist in source code or Git history, all client-facing error outputs are sanitized, file upload pipelines enforce strict size/format/signature/dimension validation, and API endpoints are shielded by HTTP security headers and rate limiting.

---

## 2. Hardening Controls & Defensive Architecture

### 2.1 Secrets Audit
- **Keyword & Regex Scan**: Audited all Python, YAML, JSON, shell, and documentation files for API keys, passwords, bearer tokens, and private keys.
- **Git History Verification**: Verified `git log` across all commits. Confirmed zero committed `.env`, `.pem`, `.key`, or credentials files.
- **Environment Variable Abstraction**: All runtime configurations (`VISIONINSPECT_ENV`, `VISIONINSPECT_CORS_ORIGINS`, `VISIONINSPECT_BACKEND_URL`) are read via environment variables with secure production defaults.

### 2.2 Error Sanitization & Information Disclosure Protection
- **No Stack Traces**: Replaced unhandled 500 error propagation with a centralized FastAPI exception handler returning clean JSON responses. Detailed tracebacks and diagnostics are logged strictly server-side (`logger.error`).
- **Path Sanitization**: Replaced local filesystem paths with logical relative identifiers (`models/{category}/patchcore_v23/`). Local user paths (`C:\Users\...`, `X:\...`, virtualenv names) are suppressed.
- **Production Mode**: When `VISIONINSPECT_ENV=production`, interactive OpenAPI documentation (`/docs`, `/redoc`) is disabled to prevent schema reconnaissance.

### 2.3 Upload Pipeline Hardening & Resource Protection
- **Size Limitation**: Maximum request size enforced at **10 MB** (`MAX_UPLOAD_SIZE_BYTES`). Excessively large files are rejected before buffer allocation.
- **Extension & Magic Byte Verification**: File extensions are whitelisted (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`). Raw file headers are checked against magic byte signatures (`\x89PNG`, `\xff\xd8\xff`, `RIFF`, `BM`) to reject disguised executables or shell scripts.
- **Decompression Bomb Protection**: Pillow's `MAX_IMAGE_PIXELS` set to **16,000,000** (16 MP). Spatial dimensions exceeding **4096 × 4096 px** are rejected prior to NumPy array conversion.
- **Image Integrity Verification**: `PIL.Image.verify()` validates stream integrity before conversion, rejecting corrupted or malformed image headers.
- **Path Traversal Shield**: Filenames are sanitized via `Path(filename).name`, preventing directory traversal attempts (`../../etc/passwd`).

### 2.4 API Security, CORS & Rate Limiting
- **CORS Restraint**: Replaced wildcard `allow_origins=["*"]` with an environment-controlled origin whitelist (`VISIONINSPECT_CORS_ORIGINS`). Defaults to known local web UI ports (`8501`, `3000`).
- **HTTP Security Headers**: Injected into every response via `SecurityHeadersMiddleware`:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Permissions-Policy: camera=(), microphone=(), geolocation=()`
  - `Content-Security-Policy: default-src 'self'; ...`
  - `Strict-Transport-Security: max-age=31536000` (on HTTPS)
- **Sliding-Window Rate Limiting**: Expensive inference routes (`/predict`, `/predict/batch`) are throttled to **60 requests/minute per client IP**. Excess traffic receives standard HTTP `429 Too Many Requests`.

### 2.5 ML Model & PyTorch Deserialization Security
- **Strict Category Whitelist**: Only the five approved categories (`bottle`, `leather`, `transistor`, `zipper`, `screw`) are accepted. Arbitrary model selection or user-directed file loading is strictly forbidden.
- **Static Model Resolution**: Model paths are constructed entirely server-side (`Path("models") / category / "patchcore_v23"`), preventing user-controlled path manipulation.
- **Safe Weight Deserialization**: PyTorch model weights load with `weights_only=True` where supported, eliminating pickle deserialization attack vectors.

### 2.6 Frontend XSS Prevention & PDF Report Security
- **HTML Sanitization**: User-controllable filenames, component categories, and detected labels passed to Streamlit modal dialogs and HTML blocks are escaped via `html.escape()`.
- **ReportLab Markup Protection**: User strings rendered in PDF flowables are sanitized through `_clean_text()` and XML entity substitution (`&amp;`, `&lt;`, `&gt;`), preventing markup injection and syntax crashes.

### 2.7 Hardened Docker Containerization
- **Minimal Debian Base**: Multi-stage build based on `python:3.12-slim`.
- **Unprivileged Runtime**: Runs as non-root user `visionuser` (UID/GID `10001`).
- **Read-Only Models**: File permissions on `./models` restricted to `550` (read & execute only).
- **Clean Image Context**: Comprehensive `.dockerignore` prevents `.git`, `.env`, tests, caches, and datasets from entering the build artifact.

---

## 3. Automated Security Verification Results

An automated security test suite (`tests/test_security_hardening.py`) was executed against the local service:

| Test Case | Attack / Stress Vector | Expected Behavior | Actual Result |
| :--- | :--- | :--- | :---: |
| `test_security_reject_executable_extension` | Upload `malicious.exe` | HTTP 400 (`Unsupported file format`) | **PASS** |
| `test_security_reject_script_extension` | Upload `exploit.sh` | HTTP 400 (`Unsupported file format`) | **PASS** |
| `test_security_reject_corrupted_data` | Non-image raw byte stream | HTTP 400 (`Invalid image file`) | **PASS** |
| `test_security_reject_empty_file` | 0-byte file upload | HTTP 400 (`Uploaded file is empty`) | **PASS** |
| `test_security_path_traversal_filename` | Filename `../../../../etc/passwd` | Filename sanitized to `passwd` | **PASS** |
| `test_security_reject_oversized_payload` | 11 MB payload stream | HTTP 400 (`Exceeds maximum allowed limit`) | **PASS** |
| `test_security_reject_decompression_bomb` | 5000 × 5000 px image | HTTP 400 (`Exceed maximum allowed dimension`) | **PASS** |
| `test_security_headers_present` | Inspect response headers | `X-Content-Type-Options`, `CSP`, `X-Frame-Options` present | **PASS** |
| `test_security_rate_limiting_enforced` | 61 rapid requests from IP | Request 61 throttled with HTTP 429 | **PASS** |
| `test_security_unsupported_category_leakage`| Request category `malicious_eval`| Clean HTTP 400, no stack trace/local path | **PASS** |
| `test_security_category_strict_mapping` | Audit `/categories` response | Strict whitelist of 5 production models | **PASS** |

### Complete Regression Suite
```text
================= 121 passed, 1 warning in 124.43s (0:02:04) ==================
```

---

## 4. Known Operational Limitations

1. **In-Memory Rate Limiting**: The built-in rate limiter runs in process memory. In a distributed multi-instance deployment behind a load balancer, an external Redis-backed rate limiter is recommended if per-instance limits are insufficient.
2. **Reverse Proxy TLS Termination**: TLS/HTTPS should be terminated at the cloud ingress/reverse proxy (Nginx, Render, AWS ALB). The backend expects `X-Forwarded-For` and `X-Forwarded-Proto` headers when running behind a proxy.
3. **OpenMP Windows Warning**: Under Windows with Intel MKL, `KMP_DUPLICATE_LIB_OK=TRUE` is set in `.env.example`. In production Linux Docker containers, this is harmlessly ignored.
