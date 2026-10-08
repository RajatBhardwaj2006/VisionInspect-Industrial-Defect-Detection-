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

---

## 7. Zero-Cost ($0 / ₹0) Cloud Deployment Options

VisionInspect is engineered to run at zero hosting cost without requiring a credit card or paid compute plan.

### Architecture Memory Analysis
- **Base PyTorch + FastAPI runtime:** ~273 MB
- **ResNet-18 Backbone:** ~45 MB
- **PatchCore v2.3 Memory Bank per Category:** ~150 MB to 235 MB
- **Single-Model Active Footprint:** ~550 MB to 640 MB (with lazy loading and chunked cdist)
- **All 5 Models Simultaneously:** ~1,290 MB (~1.3 GB)

---

### Option A: Render Cloud (Free Tier - 512 MB)
- **Cost:** $0.00 / ₹0 (No credit card required)
- **Blueprint:** `render.yaml` (`plan: free`)
- **Note:** Render's free tier imposes a strict 512 MB RAM ceiling. Because PyTorch + PatchCore memory bank for screw requires ~550–640 MB during inference, heavy inference may encounter memory pressure on Render Free.

#### Deploying on Render (Free Plan):
1. In the **Render Dashboard**, click **New +** $\rightarrow$ **Blueprint**.
2. Connect your GitHub repository:
   `https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-.git`
3. Select branch: `main`.
4. Render detects `render.yaml` with `plan: free` and launches at $0 with zero billing prompts.

---

### Option B: Streamlit Community Cloud (Recommended for $0 Hosting - 1.0 GB RAM)
- **Cost:** $0.00 / ₹0 (Free forever for GitHub repositories, zero credit card)
- **Memory Allocation:** ~1.0 GB RAM (fits single-model lazy loaded PatchCore easily)
- **Public URL:** `https://<your-app>.streamlit.app`
- **Architecture:** Streamlit native runner with automatic in-process inference fallback (`app/app.py` $\rightarrow$ `app/backend.py`).

#### Deploying to Streamlit Community Cloud in 60 Seconds:
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **Create app**.
3. Select:
   - **Repository:** `RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
4. Click **Deploy!**
5. Streamlit Community Cloud automatically installs dependencies, serves the full industrial UI, and exposes a public HTTPS URL.

