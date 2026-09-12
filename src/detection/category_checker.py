import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.patchcore_v22 import FeatureExtractorV22

SUPPORTED_CATEGORIES = ['bottle', 'leather', 'transistor', 'zipper', 'screw']

class CategoryCompatibilityChecker:
    """
    Conservative Category Compatibility Checker using PatchCore ResNet18 feature space.
    
    IMPORTANT:
    This is NOT a product classifier and does not claim to be one.
    It computes feature distances against category normal prototypes to detect whether
    an uploaded product image is markedly incompatible with the selected category model
    (e.g., uploading a zipper image when 'bottle' is selected).
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        prototype_cache_path: Optional[str] = None,
        margin_threshold: float = 0.35
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.margin_threshold = margin_threshold
        self.feature_extractor = FeatureExtractorV22(self.device)
        self.prototypes: Dict[str, torch.Tensor] = {}
        
        self.cache_path = Path(prototype_cache_path or "models/category_prototypes.pt")
        self._load_or_compute_prototypes()
        
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def _load_or_compute_prototypes(self):
        """Loads cached category prototypes or computes them from category memory banks."""
        if self.cache_path.exists():
            try:
                loaded = torch.load(self.cache_path, map_location=self.device)
                if all(cat in loaded for cat in SUPPORTED_CATEGORIES):
                    self.prototypes = {cat: loaded[cat].to(self.device) for cat in SUPPORTED_CATEGORIES}
                    return
            except Exception:
                pass
                
        # Compute from existing memory banks
        print("[CategoryCompatibilityChecker] Computing category prototypes from memory banks...")
        computed = {}
        for cat in SUPPORTED_CATEGORIES:
            mb_path = Path(f"models/{cat}/patchcore_v23/memory_bank.pt")
            if not mb_path.exists():
                raise FileNotFoundError(
                    f"Cannot initialize CategoryCompatibilityChecker: Memory bank for '{cat}' missing at {mb_path}."
                )
            mb = torch.load(mb_path, map_location="cpu")
            # The category prototype is the centroid vector in 448-dimensional feature space
            centroid = mb.mean(dim=0).to(self.device)
            computed[cat] = centroid
            
        self.prototypes = computed
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in computed.items()}, self.cache_path)
            print(f"[CategoryCompatibilityChecker] Cached category prototypes to: {self.cache_path}")
        except Exception as e:
            print(f"Warning: Could not cache category prototypes: {e}")

    def check_compatibility(
        self,
        image_input: Any,
        selected_category: str
    ) -> Dict[str, Any]:
        """
        Evaluates category compatibility for an image against the selected category.
        
        Returns:
            dict containing:
                selected_category (str)
                best_compatible_category (str)
                is_mismatch (bool)
                is_inconclusive (bool)
                confidence_margin (float)
                warning_message (Optional[str])
                scores (Dict[str, float])
        """
        selected_category = selected_category.strip().lower()
        if selected_category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"Unknown selected category: '{selected_category}'. Supported: {SUPPORTED_CATEGORIES}")
            
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            raise TypeError(f"Expected file path or PIL.Image, got {type(image_input)}")
            
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            patch_feats, _ = self.feature_extractor(img_tensor) # [1, 4096, 448]
            img_prototype = patch_feats.squeeze(0).mean(dim=0)  # [448]
            
        distances: Dict[str, float] = {}
        cosine_sims: Dict[str, float] = {}
        
        for cat, proto in self.prototypes.items():
            dist = torch.norm(img_prototype - proto, p=2).item()
            cos = F.cosine_similarity(img_prototype.unsqueeze(0), proto.unsqueeze(0)).item()
            distances[cat] = round(dist, 4)
            cosine_sims[cat] = round(cos, 4)
            
        # Determine closest matching category
        sorted_by_dist = sorted(distances.items(), key=lambda x: x[1])
        best_category, best_dist = sorted_by_dist[0]
        selected_dist = distances.get(selected_category, best_dist)
        
        # Margin: distance difference between selected and best
        confidence_margin = round(selected_dist - best_dist, 4)
        
        # Conservative mismatch decision:
        # A mismatch is flagged ONLY if:
        # 1. Best category is different from selected category.
        # 2. Confidence margin exceeds configurable threshold (default 0.35).
        is_mismatch = False
        is_inconclusive = False
        warning_message = None
        
        if best_category == selected_category:
            is_mismatch = False
            is_inconclusive = False
            warning_message = None
        elif confidence_margin >= self.margin_threshold:
            is_mismatch = True
            is_inconclusive = False
            warning_message = (
                f"Selected category: {selected_category.capitalize()}. "
                f"This image appears more compatible with: {best_category.capitalize()}."
            )
        else:
            # Small ambiguous margin: do NOT show false alarm
            is_mismatch = False
            is_inconclusive = True
            warning_message = None
            
        return {
            "selected_category": selected_category,
            "best_compatible_category": best_category,
            "is_mismatch": is_mismatch,
            "is_inconclusive": is_inconclusive,
            "confidence_margin": confidence_margin,
            "margin_threshold": self.margin_threshold,
            "warning_message": warning_message,
            "distances": distances,
            "cosine_similarities": cosine_sims
        }
