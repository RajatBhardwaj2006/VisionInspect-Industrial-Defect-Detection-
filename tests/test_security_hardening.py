"""
tests/test_security_hardening.py
================================
Defensive security suite testing VisionInspect API and upload pipeline:
1. Rejection of oversized payloads (> 10MB)
2. Rejection of unsupported file extensions (.exe, .sh, .py, etc.)
3. Rejection of corrupted / non-image byte streams
4. Rejection of empty file uploads
5. Sanitization of path traversal attempt in filename (../../etc/passwd)
6. Protection against decompression bombs (excessive pixel dimensions)
7. Production security headers verification (X-Content-Type-Options, CSP, Frame protection)
8. Rate limiting throttling under high request volume
9. Information disclosure prevention (no stack traces, no internal filesystem paths)
10. Strict category validation (rejection of unauthorized model names)
"""

import io
import time
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.backend import app, rate_limiter

@pytest.fixture
def client():
    # Clear rate limiter history for clean test runs
    rate_limiter.requests.clear()
    return TestClient(app)

def _create_image(format="PNG", size=(64, 64)):
    img = Image.new("RGB", size, color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# 1. Unsupported extensions rejection
def test_security_reject_executable_extension(client):
    buf = _create_image()
    resp = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("malicious.exe", buf, "application/octet-stream")}
    )
    assert resp.status_code == 400
    assert "unsupported file format" in resp.json()["detail"].lower()

def test_security_reject_script_extension(client):
    buf = _create_image()
    resp = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("exploit.sh", buf, "text/x-shellscript")}
    )
    assert resp.status_code == 400
    assert "unsupported file format" in resp.json()["detail"].lower()

# 2. Corrupted file rejection
def test_security_reject_corrupted_data(client):
    corrupt_bytes = io.BytesIO(b"Not an image header at all")
    resp = client.post(
        "/predict",
        data={"category": "leather"},
        files={"file": ("sample.png", corrupt_bytes, "image/png")}
    )
    assert resp.status_code == 400
    assert "invalid image file" in resp.json()["detail"].lower()

# 3. Empty file rejection
def test_security_reject_empty_file(client):
    empty_bytes = io.BytesIO(b"")
    resp = client.post(
        "/predict",
        data={"category": "screw"},
        files={"file": ("empty.png", empty_bytes, "image/png")}
    )
    assert resp.status_code == 400
    assert "uploaded file is empty" in resp.json()["detail"].lower()

# 4. Path traversal filename sanitization
def test_security_path_traversal_filename(client):
    buf = _create_image()
    traversal_name = "../../../../../etc/passwd"
    resp = client.post(
        "/predict",
        data={"category": "transistor", "return_visualizations": "false"},
        files={"file": (traversal_name, buf, "image/png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    # Ensure sanitized filename does not include traversal separators
    assert ".." not in data["filename"]
    assert "/" not in data["filename"]
    assert "\\" not in data["filename"]
    assert data["filename"] == "passwd"

# 5. Oversized upload rejection (> 10MB)
def test_security_reject_oversized_payload(client):
    # Fake PNG header followed by 11MB of payload
    huge_bytes = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024))
    resp = client.post(
        "/predict",
        data={"category": "zipper"},
        files={"file": ("huge.png", huge_bytes, "image/png")}
    )
    assert resp.status_code == 400
    assert "exceeds maximum allowed limit" in resp.json()["detail"].lower()

# 6. Decompression bomb prevention
def test_security_reject_decompression_bomb_dimensions(client):
    # 5000 x 5000 image exceeds 4096 limit
    buf = _create_image(size=(5000, 100))
    resp = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("bomb.png", buf, "image/png")}
    )
    assert resp.status_code == 400
    assert "exceed maximum allowed dimension" in resp.json()["detail"].lower()

# 7. Security Headers Verification
def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    headers = resp.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in headers

# 8. Rate Limiting verification
def test_security_rate_limiting_enforced(client):
    buf = _create_image()
    # Send requests up to limit
    for _ in range(60):
        rate_limiter.requests["testclient"].append(time.time())
        
    # The 61st request should be throttled with HTTP 429
    resp = client.post(
        "/predict",
        data={"category": "bottle"},
        files={"file": ("sample.png", buf, "image/png")}
    )
    assert resp.status_code == 429
    assert "rate limit exceeded" in resp.json()["detail"].lower()

# 9. No stack trace or internal path leakage on 404/500
def test_security_unsupported_category_leakage(client):
    buf = _create_image()
    resp = client.post(
        "/predict",
        data={"category": "malicious_eval_model"},
        files={"file": ("sample.png", buf, "image/png")}
    )
    assert resp.status_code == 400
    text = resp.text
    assert "traceback" not in text.lower()
    assert "C:\\" not in text
    assert "X:\\" not in text
    assert "/home/" not in text

# 10. Valid categories strictly mapped
def test_security_category_strict_mapping(client):
    resp = client.get("/categories")
    assert resp.status_code == 200
    data = resp.json()
    cats = [c["category"] for c in data["categories"]]
    assert set(cats) == {"bottle", "leather", "transistor", "zipper", "screw"}
    for c in data["categories"]:
        assert not c["model_dir"].startswith(("/", "\\", "C:", "X:"))
