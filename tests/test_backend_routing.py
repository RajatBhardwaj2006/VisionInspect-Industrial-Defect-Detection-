"""Exercise real category normalization and single-model cache management."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app import backend


@pytest.fixture
def detector_factory(monkeypatch, tmp_path):
    # Real directory checks and repository config, but no checkpoint reads.
    for category in ("bottle", "leather", "transistor", "zipper", "screw"):
        (tmp_path / "models" / category / "patchcore_v23").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend, "DETECTOR_CACHE", {})
    factory = Mock(side_effect=lambda category, device: SimpleNamespace(category=category, device=device))
    monkeypatch.setattr(backend, "PatchCoreDetectorV23", factory)
    return factory


@pytest.mark.parametrize("category", ["bottle", "leather", "transistor", "zipper", "screw"])
def test_normalized_category_routes_to_its_detector(detector_factory, category):
    detector = backend.get_detector(f"  {category.upper()}  ")
    assert detector.category == category
    assert backend.DETECTOR_CACHE == {category: detector}
    detector_factory.assert_called_once_with(category=category, device=backend.DEVICE)


def test_repeated_category_reuses_cached_detector(detector_factory):
    first = backend.get_detector("bottle")
    assert backend.get_detector(" BOTTLE ") is first
    assert detector_factory.call_count == 1


def test_switching_category_evicts_previous_detector(detector_factory):
    first = backend.get_detector("bottle")
    second = backend.get_detector("screw")
    assert second is not first
    assert backend.DETECTOR_CACHE == {"screw": second}
    assert backend.get_detector("bottle") is not first
    assert list(backend.DETECTOR_CACHE) == ["bottle"]
    assert detector_factory.call_count == 3


@pytest.mark.parametrize("category", ["wood", "../bottle", "models/bottle", ""])
def test_unsupported_category_cannot_construct_or_evict_model(detector_factory, category):
    active = backend.get_detector("bottle")
    detector_factory.reset_mock()
    with pytest.raises(HTTPException) as exc:
        backend.get_detector(category)
    assert exc.value.status_code == 400
    assert backend.DETECTOR_CACHE == {"bottle": active}
    detector_factory.assert_not_called()


def test_missing_model_directory_preserves_active_cache(detector_factory, tmp_path):
    active = backend.get_detector("bottle")
    (tmp_path / "models" / "screw" / "patchcore_v23").rmdir()
    detector_factory.reset_mock()
    with pytest.raises(HTTPException) as exc:
        backend.get_detector("screw")
    assert exc.value.status_code == 404
    assert "unavailable" in exc.value.detail
    assert str(tmp_path) not in exc.value.detail
    assert backend.DETECTOR_CACHE == {"bottle": active}
    detector_factory.assert_not_called()
