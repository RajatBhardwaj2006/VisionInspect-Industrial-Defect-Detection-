"""HTTP boundary tests without loading checkpoints or running neural inference."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import backend


@pytest.fixture
def client(monkeypatch):
    # Keep real routes, upload validation and response assembly; replace model I/O.
    detector = SimpleNamespace(inspect=Mock(side_effect=RuntimeError("private-model-diagnostic")))
    monkeypatch.setattr(backend, "get_detector", lambda category: detector)
    checker = SimpleNamespace(check_compatibility=Mock(return_value={"is_mismatch": False}))
    monkeypatch.setattr(backend, "get_compatibility_checker", lambda: checker)
    monkeypatch.setattr(backend, "rate_limiter", backend.InMemoryRateLimiter())
    with TestClient(backend.app) as test_client:
        yield test_client, detector


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize("route,field", [("/predict", "file"), ("/predict/batch", "files")])
@pytest.mark.parametrize("body", [{}, {"category": "bottle", "file": "not-an-upload"}])
def test_missing_multipart_upload_returns_validation_error(client, route, field, body):
    test_client, detector = client
    response = test_client.post(route, json=body)
    assert response.status_code == 422
    assert any(error["loc"][-1] == field for error in response.json()["detail"])
    detector.inspect.assert_not_called()


@pytest.mark.parametrize("route,field", [("/predict", "file"), ("/predict/batch", "files")])
def test_invalid_boolean_form_field_does_not_invoke_inference(client, route, field):
    test_client, detector = client
    response = test_client.post(
        route, data={"return_visualizations": "perhaps"},
        files={field: ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 422
    assert any(error["loc"][-1] == "return_visualizations" for error in response.json()["detail"])
    detector.inspect.assert_not_called()


def test_single_inference_failure_returns_sanitized_500(client):
    test_client, detector = client
    response = test_client.post(
        "/predict", files={"file": ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "An error occurred during defect inspection."}
    assert "private-model-diagnostic" not in response.text
    detector.inspect.assert_called_once()


def test_batch_inference_failure_does_not_abort_later_images(client):
    test_client, detector = client
    detector.inspect.side_effect = [RuntimeError("private-model-diagnostic"), {
        "status": "NORMAL", "score": 0.5, "image_threshold": 1.5,
        "pixel_threshold": 1.4, "localized_regions": [],
        "explanation": "No significant anomaly detected.",
    }]
    response = test_client.post("/predict/batch", files=[
        ("files", ("first.png", png_bytes(), "image/png")),
        ("files", ("empty.png", b"", "image/png")),
        ("files", ("last.png", png_bytes(), "image/png")),
    ])
    assert response.status_code == 200
    payload = response.json()
    assert [result["filename"] for result in payload["results"]] == ["first.png", "empty.png", "last.png"]
    assert [result.get("error") for result in payload["results"]] == [
        "Inspection failed for this image.", "Uploaded file is empty.", None,
    ]
    assert payload["summary"] == {"total": 3, "normal": 1, "defective": 0, "category_warnings": 0, "failed": 2}
    assert [result["status"] for result in payload["results"]] == ["ERROR", "ERROR", "NORMAL"]
    assert payload["results"][-1]["anomaly_score"] == 0.5
    assert "private-model-diagnostic" not in response.text
    assert detector.inspect.call_count == 2


def test_corrupt_png_with_valid_signature_is_rejected_before_inference(client):
    test_client, detector = client
    response = test_client.post("/predict", files={
        "file": ("truncated.png", b"\x89PNG\r\n\x1a\ntruncated", "image/png"),
    })
    assert response.status_code == 400
    assert "Corrupted or unreadable" in response.json()["detail"]
    detector.inspect.assert_not_called()
