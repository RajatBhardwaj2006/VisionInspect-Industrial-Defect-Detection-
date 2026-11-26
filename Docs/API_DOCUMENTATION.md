# FastAPI reference

Run from the repository root in an activated environment:

```powershell
python -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
```

Base URL: `http://127.0.0.1:8000`. The routes below come from
[`app/backend.py`](../app/backend.py). `/docs` and `/redoc` are enabled unless
`VISIONINSPECT_ENV=production`; `/openapi.json` remains available. The root
response still advertises `/docs` when the interactive docs are disabled.

## Read-only endpoints

| Method | Path | Response |
| --- | --- | --- |
| GET | `/` | Service name, `status: running`, version and links to health, categories and docs |
| GET | `/health` | `status`, `supported_categories`, `loaded_models`, `device`, `version` |
| GET | `/categories` | `categories` array: category, status, thresholds, spatial-prior flag, backbone and logical model directory |

Example health response (device and loaded models depend on runtime):

```json
{"status":"ok","supported_categories":["bottle","leather","transistor","zipper","screw"],"loaded_models":[],"device":"cpu","version":"3.2.0"}
```

Category `status` is `trained` when the model directory exists and `untrained`
otherwise. This is not a checkpoint integrity check. Health does not load models.

## POST /predict

Send **multipart/form-data**, with these fields:

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `file` | Yes | None | Image upload |
| `category` | No | `bottle` | `bottle`, `leather`, `transistor`, `zipper`, `screw`; whitespace stripped and case normalized |
| `return_visualizations` | No | `true` | Boolean controlling fallback reading of a saved visualization |

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "category=bottle" -F "return_visualizations=false" -F "file=@assets/test_samples/bottle/broken_large_000.png"
```

On macOS/Linux use `curl` in place of `curl.exe`. A successful response is 200.
Illustrative excerpt (values are not a benchmark prediction):

```json
{
  "request_id": "req_<timestamp>_<id>",
  "filename": "sample.png",
  "category": "bottle",
  "model": "PatchCore v2.3",
  "model_path": "models/bottle/patchcore_v23/",
  "status": "NORMAL",
  "is_defective": false,
  "anomaly_score": 0.8,
  "image_threshold": 1.5,
  "pixel_threshold": 1.4,
  "decision_margin": -0.7,
  "num_defects": 0,
  "localized_regions": [],
  "defect_regions": []
}
```

Additional fields are `peak_anomaly`, `memory_bank`, `heatmap_base64`,
`visualization_base64`, `inference_time_ms`, `inference_time_s`, `explanation`,
`why_explanation`, `category_compatibility`, `compatibility` and `category_warning`.
The two region arrays are aliases; each region has `id`, `label`, `intensity`,
`bbox: [x, y, width, height]`, `area`, `score` and `aspect_ratio`. Compatibility
payloads are aliases too. A mismatch warns without rerouting the request.

`return_visualizations=false` does not guarantee that base64 fields are absent:
the response still forwards visualizations already returned by the detector.

## POST /predict/batch

Use **1–5** repeated `files` fields, with the same optional category and
visualization fields. One category applies to the whole batch.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/batch -F "category=bottle" -F "files=@assets/test_samples/bottle/broken_large_000.png" -F "files=@assets/test_samples/bottle/good_001.png"
```

The 200 envelope contains `category`, `model`, `model_path`, `batch_size`,
`total_images`, `summary` and ordered `results`. Successful entries use the
single-image schema. A failed entry is, for example:

```json
{"filename":"empty.png","status":"ERROR","is_defective":false,"error":"Uploaded file is empty."}
```

`summary` counts `total`, `normal`, `defective`, `category_warnings` and `failed`.
A batch can return **200 with failed entries**, even if every image fails.
`is_defective: false` on an error entry is not a passing inspection; check `status`.

## Validation and errors

Uploads allow `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` (case insensitive), up to
10 MiB, width/height at most 4096 and total pixels at most 16,000,000. Validation
checks a recognized signature, image decoding and dimensions. An extensionless
filename is allowed when content decodes; the implementation does not require
the extension to match the decoded format. Declared MIME type alone is not proof
of validity. Images are converted to RGB and resized by the detector.

| HTTP code | Trigger | Response shape |
| --- | --- | --- |
| 400 | Unsupported category, invalid single upload, more than five batch files | `{"detail":"..."}` |
| 404 | Supported category's model directory missing | `{"detail":"Inspection model for category '<category>' is currently unavailable."}` |
| 422 | Missing required upload, JSON instead of multipart, invalid boolean field | `{"detail":[...]}` validation locations and messages |
| 429 | Inference-route request limit exceeded | `{"detail":"Rate limit exceeded. Maximum 60 requests per minute allowed."}` |
| 500 | Single-image inspection exception | `{"detail":"An error occurred during defect inspection."}` |
| 500 | Other unhandled exception | Generic internal-inspection error; details stay in server logs |

Detector/checker initialization happens before single-upload validation; missing
or unreadable model artifacts can therefore fail before image errors are returned.
Batch per-image validation and inference exceptions become `ERROR` entries instead
of HTTP 400/500. Missing batch `files` normally produces framework HTTP 422.

The process-local limiter allows 60 requests per 60 seconds per client on
`/predict` and `/predict/batch`, including invalid requests. A batch consumes one
request. The middleware reads `X-Forwarded-For` when supplied; deploy behind a
proxy that controls that header if relying on client-IP limits. CORS origins come
from `VISIONINSPECT_CORS_ORIGINS` or local development defaults. These routes do
not implement API-key authentication.
