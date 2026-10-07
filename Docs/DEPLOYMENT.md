# VisionInspect — Production Deployment Guide

> **Document Type:** Production Readiness, Networking, and Deployment Architecture  
> **Target Environment:** On-Premise Industrial Server, Edge IPC (Windows/Linux), Docker Container  
> **Version:** 3.2.0

---

## 1. Architecture Topology

```
             ┌──────────────────────────────────────────┐
             │       Reverse Proxy / Gateway (Nginx)    │
             └──────┬────────────────────────────┬──────┘
                    │ Port 8501 (WebSocket/HTTP) │ Port 8000 (REST)
                    ▼                            ▼
      ┌──────────────────────────┐  ┌──────────────────────────┐
      │   Streamlit Frontend     │  │     FastAPI Backend      │
      │   (app/app.py)           │  │     (app/backend.py)     │
      └──────────────────────────┘  └────────────┬─────────────┘
                                                 │
                                                 ▼
                                    ┌──────────────────────────┐
                                    │ PyTorch / TorchVision    │
                                    │ Coreset Memory Banks     │
                                    │ (models/<category>/)     │
                                    └──────────────────────────┘
```

---

## 2. Environment Variables

Create `.env` based on `.env.example`:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `VISIONINSPECT_BACKEND_URL` | `http://127.0.0.1:8000` | Address where the frontend sends inspection requests |
| `VISIONINSPECT_HOST` | `127.0.0.1` | Bind host for backend service |
| `VISIONINSPECT_PORT` | `8000` | Port for backend REST API |
| `VISIONINSPECT_FRONTEND_PORT` | `8501` | Port for Streamlit Web UI |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | Resolves OpenMP DLL conflicts on Windows Intel CPUs |

---

## 3. Production Service Startup

### Backend Service (FastAPI)
Run with production Uvicorn settings:
```powershell
python -m uvicorn app.backend:app --host 0.0.0.0 --port 8000 --workers 1
```
> **Note on Workers:** Because PatchCore models keep memory bank tensors in memory (~1.2 GB RAM for all 5 categories combined), use `--workers 1` or `--workers 2` depending on available system RAM to prevent excessive duplicate memory usage.

### Frontend Service (Streamlit)
Run with production Streamlit flags:
```powershell
python -m streamlit run app/app.py --server.port 8501 --server.headless true --server.enableCORS true
```

---

## 4. Model Availability Checklist

Before launching production services, verify:
1. `git lfs pull` has been executed to download production memory banks.
2. Run `python scripts/verify_models.py` — it must report `[STATUS: READY]`.
3. The server has at least **4 GB RAM** (for CPU inference) or **4 GB VRAM** (for CUDA inference).

---

## 5. Health Monitoring

Monitor the REST API via:
- Endpoint: `GET http://127.0.0.1:8000/health`
- Expected payload:
  ```json
  {
    "status": "ok",
    "supported_categories": ["bottle", "leather", "transistor", "zipper", "screw"],
    "loaded_models": ["bottle", "leather"],
    "device": "cpu",
    "version": "3.2.0"
  }
  ```

---

## 6. PDF Generation System

PDF inspection certificates are generated in-memory using **ReportLab Platypus**:
- Dependency: `reportlab>=4.0.0`
- Zero temporary disk writes: Reports are compiled directly to `io.BytesIO` streams and served via base64 or Streamlit download buttons.
- No external PDF rendering engines (e.g. wkhtmltopdf) are required.
