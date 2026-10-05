"""
VisionInspect - Feature Extraction Backbones
Provides convenience imports for PatchCore feature extractors.
"""
from src.models.patchcore import FeatureExtractor
from src.models.patchcore_v22 import FeatureExtractorV22

__all__ = ["FeatureExtractor", "FeatureExtractorV22"]
