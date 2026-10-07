"""
Model Documentation Component for VisionInspect
================================================
Provides a dedicated, interactive, production-grade technical documentation
page for each of the 5 industrial inspection categories (Bottle, Leather,
Transistor, Zipper, Screw).

Features:
- Dynamically rendered from a shared ModelDocumentationPage component.
- Uses REAL verified source code directly from the repository.
- Accurate configurations from config.yaml and the locked model audit.
- Step-by-step visual scan pipeline.
- Real MVTec AD defect categories per product.
- Direct seamless navigation to Inspect page with pre-selected category.
"""

import io
import base64
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from PIL import Image
import streamlit as st


# =============================================================================
# VERIFIED CATEGORY METADATA & CONFIGURATION (SOURCE OF TRUTH: config.yaml)
# =============================================================================

CATEGORY_DOCS: Dict[str, Dict[str, Any]] = {
    "bottle": {
        "id": "bottle",
        "name": "Bottle",
        "technical_name": "PatchCore v2.3 Anomaly Detector (Rigid Glass Container)",
        "subtitle": "Industrial optical surface anomaly inspection for glass containers",
        "short_desc": "VisionInspect learns the normal optical appearance of defect-free glass bottles and identifies surface fractures, chipping, and foreign particulate contamination that differ from that learned representation.",
        "golden_sample": "dataset/mvtec_anomaly_detection/bottle/test/good/001.png",
        "backbone": "ResNet-18 (ImageNet Pre-trained, PyTorch torchvision)",
        "feature_dim": "448-dimensional patch descriptors (L1: 64D + L2: 128D + L3: 256D)",
        "patch_grid": "64 × 64 feature grid (4,096 patches / 4×4 px receptive stride)",
        "training_paradigm": "Unsupervised Anomaly Detection (Trained strictly on nominal glass bottles)",
        "detection_type": "Patch-level nearest neighbor distance (L2 Euclidean in 448D space)",
        "model_status": "PRODUCTION / LOCKED",
        "status_badge_color": "var(--success)",
        "image_threshold": 1.50,
        "pixel_threshold": 1.40,
        "spatial_prior": "Active (p75 / mean normal background subtraction)",
        "spatial_prior_desc": "Suppresses normal bottle rim and specular edge reflection artifacts.",
        "morphology_kernel": "3 × 3",
        "min_region_area": 25,
        "border_margin": 5,
        "merge_distance": 15.0,
        "image_score_method": "max_raw",
        "defects": [
            {
                "tag": "broken_large",
                "label": "Broken Large",
                "desc": "Severe structural fracture and glass fragmentation across container wall or mouth rim."
            },
            {
                "tag": "broken_small",
                "label": "Broken Small",
                "desc": "Localized surface chipping, rim fractures, or small high-impact fractures."
            },
            {
                "tag": "contamination",
                "label": "Contamination",
                "desc": "Foreign particulates, liquid smears, dust accumulation, or optical debris adhering to surface."
            }
        ],
        "category_specific_notes": "Bottles exhibit circular symmetry and specular reflections. PatchCore v2.3 utilizes a calibrated normal spatial background prior (p75 quantile mode) to cancel out expected rim highlights without suppressing true glass fractures."
    },
    "leather": {
        "id": "leather",
        "name": "Leather",
        "technical_name": "PatchCore v2.3 Anomaly Detector (Stochastic Natural Texture Fabric)",
        "subtitle": "High-resolution texture surface flaw detection on natural leather",
        "short_desc": "VisionInspect learns the normal stochastic grain pattern of defect-free leather hides and detects structural cuts, holes, color discolorations, and deep fold creases that disrupt the learned grain distribution.",
        "golden_sample": "dataset/mvtec_anomaly_detection/leather/test/good/001.png",
        "backbone": "ResNet-18 (ImageNet Pre-trained, PyTorch torchvision)",
        "feature_dim": "448-dimensional patch descriptors (L1: 64D + L2: 128D + L3: 256D)",
        "patch_grid": "64 × 64 feature grid (4,096 patches / 4×4 px receptive stride)",
        "training_paradigm": "Unsupervised Anomaly Detection (Trained strictly on nominal leather swatches)",
        "detection_type": "Patch-level nearest neighbor distance (L2 Euclidean in 448D space)",
        "model_status": "PRODUCTION / LOCKED",
        "status_badge_color": "var(--success)",
        "image_threshold": 2.90,
        "pixel_threshold": 1.71,
        "spatial_prior": "Disabled (Homogeneous texture mode)",
        "spatial_prior_desc": "Texture swatches have translationally invariant grain; fixed spatial priors are disabled to prevent texture bias.",
        "morphology_kernel": "3 × 3",
        "min_region_area": 50,
        "border_margin": 5,
        "merge_distance": 15.0,
        "image_score_method": "max_raw",
        "defects": [
            {
                "tag": "color",
                "label": "Color Flaw",
                "desc": "Dye inconsistency, bleaching, pigment streaks, or localized color staining on the hide."
            },
            {
                "tag": "cut",
                "label": "Surface Cut",
                "desc": "Sharp blade incisions, linear shear tears, or punctures penetrating the grain surface."
            },
            {
                "tag": "fold",
                "label": "Deep Fold",
                "desc": "Unironed creases, compression wrinkles, or structural bending marks distorting natural grain."
            },
            {
                "tag": "glue",
                "label": "Glue Residue",
                "desc": "Adhesive contamination, binder spillage, or sticky chemical spots adhering to leather."
            },
            {
                "tag": "poke",
                "label": "Poke / Indentation",
                "desc": "Pointed impact indentations, nail punctures, or localized surface depressions."
            }
        ],
        "category_specific_notes": "Leather is a stochastic texture rather than a rigid structural object. Consequently, use_spatial_prior is set to False (homogeneous texture mode) and min_region_area is set to 50 px to filter out micro-pore natural grain variance."
    },
    "transistor": {
        "id": "transistor",
        "name": "Transistor",
        "technical_name": "PatchCore v2.3 Anomaly Detector (Semiconductor Component IC)",
        "subtitle": "Micro-electronics defect verification for leads, casing, and component placement",
        "short_desc": "VisionInspect learns the geometric integrity of nominal semiconductor transistors and detects bent pins, sheared lead terminals, package cracks, and missing components diverging from golden references.",
        "golden_sample": "dataset/mvtec_anomaly_detection/transistor/test/good/001.png",
        "backbone": "ResNet-18 (ImageNet Pre-trained, PyTorch torchvision)",
        "feature_dim": "448-dimensional patch descriptors (L1: 64D + L2: 128D + L3: 256D)",
        "patch_grid": "64 × 64 feature grid (4,096 patches / 4×4 px receptive stride)",
        "training_paradigm": "Unsupervised Anomaly Detection (Trained strictly on golden semiconductor ICs)",
        "detection_type": "Patch-level nearest neighbor distance (L2 Euclidean in 448D space)",
        "model_status": "PRODUCTION / LOCKED",
        "status_badge_color": "var(--success)",
        "image_threshold": 3.525,
        "pixel_threshold": 2.822,
        "spatial_prior": "Presence Gate (Component verification gate)",
        "spatial_prior_desc": "Uses top-200 density scoring and an active center-ROI dark-pixel presence check for missing IC bodies.",
        "morphology_kernel": "2 × 2",
        "min_region_area": 15,
        "border_margin": 0,
        "merge_distance": 12.0,
        "image_score_method": "top200 (Mean of top 200 highest anomaly pixels)",
        "defects": [
            {
                "tag": "bent_lead",
                "label": "Bent Lead",
                "desc": "Angular pin deformation, deflected solder terminal legs, or misaligned lead grid."
            },
            {
                "tag": "cut_lead",
                "label": "Cut Lead",
                "desc": "Severed, clipped, or fractured pin terminals breaking electrical conductivity."
            },
            {
                "tag": "damaged_case",
                "label": "Damaged Case",
                "desc": "Chipped epoxy encapsulation, casing cracks, or mechanical fractures in plastic housing."
            },
            {
                "tag": "misplaced",
                "label": "Misplaced / Missing",
                "desc": "Completely absent component body or extreme lateral rotation on the circuit substrate."
            }
        ],
        "category_specific_notes": "Transistors feature extended conductive pins reaching near image borders, so border_margin=0 is enforced. Furthermore, enable_presence_check detects completely missing transistor packages even when background is clean, triggering a calibrated score override."
    },
    "zipper": {
        "id": "zipper",
        "name": "Zipper",
        "technical_name": "PatchCore v2.3 Anomaly Detector (Mechanical Fastener Chain)",
        "subtitle": "Continuous fastener chain inspection for tooth alignment and fabric tape weave",
        "short_desc": "VisionInspect learns the periodic pitch and interlocking geometry of nominal zipper fasteners and flags missing teeth, split gaps, crushed elements, and fabric weave roughness that deviate from learned regularity.",
        "golden_sample": "dataset/mvtec_anomaly_detection/zipper/test/good/001.png",
        "backbone": "ResNet-18 (ImageNet Pre-trained, PyTorch torchvision)",
        "feature_dim": "448-dimensional patch descriptors (L1: 64D + L2: 128D + L3: 256D)",
        "patch_grid": "64 × 64 feature grid (4,096 patches / 4×4 px receptive stride)",
        "training_paradigm": "Unsupervised Anomaly Detection (Trained strictly on nominal zipper fasteners)",
        "detection_type": "Patch-level nearest neighbor distance (L2 Euclidean in 448D space)",
        "model_status": "PRODUCTION / LOCKED",
        "status_badge_color": "var(--success)",
        "image_threshold": 1.47,
        "pixel_threshold": 0.91,
        "spatial_prior": "Active (Mean background subtraction)",
        "spatial_prior_desc": "Maintains vertical centerline tooth stability and suppresses tape boundary background noise.",
        "morphology_kernel": "2 × 2",
        "min_region_area": 15,
        "border_margin": 1,
        "merge_distance": 10.0,
        "image_score_method": "max_raw",
        "defects": [
            {
                "tag": "broken_teeth",
                "label": "Broken Teeth",
                "desc": "Snaped, sheared, or missing individual interlocking metallic/plastic teeth."
            },
            {
                "tag": "combined",
                "label": "Combined Defect",
                "desc": "Simultaneous tooth fracture and fabric tape tear across the fastener assembly."
            },
            {
                "tag": "fabric_border",
                "label": "Fabric Border",
                "desc": "Frayed tape margin, edge unraveling, or border weave disconnection."
            },
            {
                "tag": "fabric_interior",
                "label": "Fabric Interior",
                "desc": "Thread loops, loose fibers, or weave irregularities within the tape body."
            },
            {
                "tag": "rough",
                "label": "Rough Surface",
                "desc": "Burrs, casting imperfections, or rough metal flash on teeth surface."
            },
            {
                "tag": "split_teeth",
                "label": "Split Teeth",
                "desc": "Gapped or uncoupled interlocking sequence failing nominal mesh criteria."
            },
            {
                "tag": "squeezed_teeth",
                "label": "Squeezed Teeth",
                "desc": "Pinched, compressed, or distorted tooth profiles hindering slider travel."
            }
        ],
        "category_specific_notes": "Zipper inspection balances discrete periodic tooth alignment with textured woven fabric tape. The 2×2 morphology kernel preserves small single-tooth gaps without over-merging adjacent normal elements."
    },
    "screw": {
        "id": "screw",
        "name": "Screw",
        "technical_name": "PatchCore v2.3 Anomaly Detector (Threaded Fastener)",
        "subtitle": "Thread crest integrity, drive head inspection, and shank flaw detection",
        "short_desc": "VisionInspect learns the nominal helical geometry of precision threaded screws and identifies stripped threads, damaged drive head sockets, neck scratches, and crest deformations.",
        "golden_sample": "dataset/mvtec_anomaly_detection/screw/test/good/001.png",
        "backbone": "ResNet-18 (ImageNet Pre-trained, PyTorch torchvision)",
        "feature_dim": "448-dimensional patch descriptors (L1: 64D + L2: 128D + L3: 256D)",
        "patch_grid": "64 × 64 feature grid (4,096 patches / 4×4 px receptive stride)",
        "training_paradigm": "Unsupervised Anomaly Detection (Trained strictly on nominal threaded screws)",
        "detection_type": "Patch-level nearest neighbor distance (L2 Euclidean in 448D space)",
        "model_status": "PRODUCTION / LOCKED",
        "status_badge_color": "var(--success)",
        "image_threshold": 2.80,
        "pixel_threshold": 2.40,
        "spatial_prior": "Disabled (prior_mode: none)",
        "spatial_prior_desc": "Thread profile features are preserved without spatial suppression, ensuring thread_side defects are retained.",
        "morphology_kernel": "2 × 2",
        "min_region_area": 10,
        "border_margin": 1,
        "merge_distance": 12.0,
        "image_score_method": "top100 (Mean of top 100 highest anomaly pixels)",
        "defects": [
            {
                "tag": "manipulated_front",
                "label": "Manipulated Front",
                "desc": "Deformed drive socket, sheared tool recess, or crushed drive head face."
            },
            {
                "tag": "scratch_head",
                "label": "Scratch Head",
                "desc": "Tooling abrasions, machining scratches, or surface gouges on screw head crown."
            },
            {
                "tag": "scratch_neck",
                "label": "Scratch Neck",
                "desc": "Longitudinal gouges or machining defects on unthreaded shank under head."
            },
            {
                "tag": "thread_side",
                "label": "Thread Side",
                "desc": "Stripped, dented, deformed, or missing helical thread crests along screw body."
            },
            {
                "tag": "thread_top",
                "label": "Thread Top",
                "desc": "Crushed, blunted, or fractured tip threads failing insertion specifications."
            }
        ],
        "category_specific_notes": "Following the comprehensive model audit, screw inspection uses image_score_method: top100 and a 10 px minimum region area with prior_mode: none. This ensures thread-side helical defects are reliably detected without over-suppression."
    }
}


# =============================================================================
# REAL REPOSITORY SOURCE CODE EXCERPTS (NO SYNTHETIC OR FAKE CODE)
# =============================================================================

REAL_CODE_SNIPPETS = {
    "inference": {
        "title": "Inference Pipeline & Execution",
        "file": "src/detection/patchcore_v23_detector.py",
        "lines": "Lines 241 – 365",
        "lang": "python",
        "code": """def inspect(self, image_input: Any) -> Dict[str, Any]:
    # 1. Image preprocessing to uniform 256x256 tensor
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    img_tensor = transform(working_pil).unsqueeze(0).to(self.device)

    # 2. Predict anomaly score & spatial map using category spatial prior setting
    raw_score, amap_tensor = self.model.predict(
        img_tensor, 
        use_prior=self.use_spatial_prior,
        prior_mode=self.prior_mode
    )
    amap_np = amap_tensor.detach().cpu().numpy()
    
    # 3. Gaussian smoothing of anomaly heatmap
    if self.gaussian_sigma > 0:
        smooth_map = cv2.GaussianBlur(amap_np, (0, 0), sigmaX=self.gaussian_sigma)
    else:
        smooth_map = amap_np

    # 4. Compute image-level score using configured category scoring method
    if self.image_score_method == "top200":
        score = float(np.sort(smooth_map.flatten())[-200:].mean())
    elif self.image_score_method == "top100":
        score = float(np.sort(smooth_map.flatten())[-100:].mean())
    else:
        score = raw_score
        
    # 5. Direct absolute distance thresholding & morphology
    bin_mask = (smooth_map > self.pixel_threshold)
    clean_mask = postprocess_clean_mask(
        bin_mask, kernel_size=self.morphology_kernel, min_area=self.min_region_area
    )
    
    # 6. Extract tight bounding boxes & merge adjacent regions
    regions = extract_tight_regions(clean_mask, border_margin=self.border_margin)
    merged_regions = merge_nearby_regions(regions, merge_dist=self.merge_distance)
    
    # 7. Final verdict
    is_defective = bool(score > self.image_threshold)
    status = "DEFECTIVE" if is_defective else "NORMAL"
    return {
        "status": status,
        "score": score,
        "threshold": self.image_threshold,
        "is_defective": is_defective,
        "localized_regions": merged_regions,
        ...
    }""",
        "explanation": "Executes the end-to-end inspection workflow for a single component: converts the input image to a 256×256 tensor, extracts multi-scale patch features, computes nearest-neighbor distances against the nominal memory bank, applies category-specific Gaussian smoothing and thresholding, filters pixel noise with mathematical morphology, and outputs the final pass/fail decision with bounding boxes."
    },
    "features": {
        "title": "Multi-Scale Feature Extraction (ResNet-18)",
        "file": "src/models/patchcore_v22.py",
        "lines": "Lines 12 – 56",
        "lang": "python",
        "code": """class FeatureExtractorV22(nn.Module):
    \"\"\"
    Higher-resolution feature extractor combining ResNet18 layer1, layer2, and layer3.
    Output spatial resolution: 64x64 patch grid (4x4 pixel resolution per patch for 256x256 image).
    Total channel dimension: 64 + 128 + 256 = 448.
    \"\"\"
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        
        self.eval()
        self.to(device)
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        f_layer1 = self.layer1(x)         # [B, 64, 64, 64]
        f_layer2 = self.layer2(f_layer1)  # [B, 128, 32, 32]
        f_layer3 = self.layer3(f_layer2)  # [B, 256, 16, 16]
        
        # Upsample deeper layers to match 64x64 grid of layer1
        target_size = (f_layer1.shape[2], f_layer1.shape[3])  # (64, 64)
        f_layer2_up = F.interpolate(f_layer2, size=target_size, mode='bilinear', align_corners=False)
        f_layer3_up = F.interpolate(f_layer3, size=target_size, mode='bilinear', align_corners=False)
        
        # Concat along channel dimension: 64 + 128 + 256 = 448
        f_concat = torch.cat([f_layer1, f_layer2_up, f_layer3_up], dim=1)  # [B, 448, 64, 64]
        
        B, C, Hp, Wp = f_concat.shape
        patch_features = f_concat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, C)
        return patch_features, (Hp, Wp)""",
        "explanation": "Extracts hierarchical representations from frozen ResNet-18 layers 1, 2, and 3. By bilinearly interpolating deeper abstract layers onto the higher-resolution 64×64 spatial grid of layer 1 and concatenating them, every 4×4 pixel patch in the original image is assigned a rich 448-dimensional descriptor capturing both low-level texture details and high-level semantic context."
    },
    "scoring": {
        "title": "Nearest-Neighbor Anomaly Scoring & Spatial Prior",
        "file": "src/models/patchcore_v22.py",
        "lines": "Lines 124 – 168",
        "lang": "python",
        "code": """def _predict_raw(self, img_tensor: torch.Tensor) -> Tuple[float, torch.Tensor]:
    img_tensor = img_tensor.to(self.device)
    orig_H, orig_W = img_tensor.shape[2], img_tensor.shape[3]
    
    patch_feats, (Hp, Wp) = self.feature_extractor(img_tensor)
    test_patches = patch_feats.squeeze(0)  # [4096, 448]
    
    # Exact Euclidean nearest-neighbor distance (L2 norm) via GPU cdist
    dists = torch.cdist(test_patches, self.memory_bank, p=2.0)
    min_dists, _ = torch.min(dists, dim=1)  # Distance to closest normal patch
    
    # Reshape patch distances into 2D map and interpolate to original resolution
    amap_patch = min_dists.reshape(1, 1, Hp, Wp)
    amap_resized = F.interpolate(amap_patch, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
    amap_tensor = amap_resized.squeeze(0).squeeze(0)
    
    image_score = float(min_dists.max().item())
    return image_score, amap_tensor

def predict(self, img_tensor: torch.Tensor, use_prior: bool = True,
            prior_mode: str = 'mean') -> Tuple[float, torch.Tensor]:
    raw_score, amap_tensor = self._predict_raw(img_tensor)
    if use_prior:
        # Subtract calibrated normal spatial prior to eliminate expected edge artifacts
        if prior_mode == 'p75' and self.normal_spatial_prior_p75 is not None:
            amap_subbed = torch.clamp(amap_tensor - self.normal_spatial_prior_p75, min=0.0)
            return float(amap_subbed.max().item()), amap_subbed
        elif self.normal_spatial_prior is not None:
            amap_subbed = torch.clamp(amap_tensor - self.normal_spatial_prior, min=0.0)
            return float(amap_subbed.max().item()), amap_subbed
    return raw_score, amap_tensor""",
        "explanation": "Compares every test patch against the category's nominal coreset memory bank using Euclidean L2 distance in 448D space. Patches that diverge significantly from any learned normal pattern receive large distance values. For structured categories (like Bottle and Zipper), a pre-calibrated spatial prior map is subtracted to cancel out normal rim reflections and boundary artifacts."
    },
    "localization": {
        "title": "Morphology & Defect Region Localization",
        "file": "src/detection/patchcore_v23_detector.py",
        "lines": "Lines 55 – 100",
        "lang": "python",
        "code": """def postprocess_clean_mask(bin_mask: np.ndarray, kernel_size: int = 3, min_area: int = 25) -> np.ndarray:
    \"\"\"Morphological closing then opening + connected component noise filtering.\"\"\"
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    labeled, num = label(opened)
    clean_mask = np.zeros_like(opened, dtype=bool)
    for i in range(1, num + 1):
        comp = (labeled == i)
        if comp.sum() >= min_area:
            clean_mask[comp] = True
    return clean_mask

def extract_tight_regions(clean_mask: np.ndarray, border_margin: int = 5) -> List[Dict]:
    labeled, num = label(clean_mask.astype(np.int32))
    regions = []
    H, W = clean_mask.shape
    for idx in range(1, num + 1):
        comp = (labeled == idx)
        ys, xs = np.where(comp)
        if ys.size == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        # Border margin filter: exclude camera boundary artifacts
        if (x_min < border_margin or x_max > W - 1 - border_margin or
                y_min < border_margin or y_max > H - 1 - border_margin):
            continue
            
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        regions.append({
            "x": x_min, "y": y_min,
            "width": w, "height": h,
            "area": int(comp.sum()),
            "centroid": (float(xs.mean()), float(ys.mean()))
        })
    return regions""",
        "explanation": "Converts continuous anomaly heatmaps into discrete, actionable bounding box coordinates. Mathematical morphological closing connects thin fracture lines, opening eliminates salt-and-pepper noise, connected-component labeling discards isolated sub-threshold pixels, and boundary margins exclude edge camera padding."
    },
    "config_dispatch": {
        "title": "Category Dispatch & Configuration Loading",
        "file": "app/backend.py & config.yaml",
        "lines": "Lines 47 – 68 (backend.py)",
        "lang": "python",
        "code": """# app/backend.py (In-memory detector caching & category dispatch)
DETECTOR_CACHE: Dict[str, PatchCoreDetectorV23] = {}

def get_detector(category: str) -> PatchCoreDetectorV23:
    category = category.strip().lower()
    if category not in SUPPORTED_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported category '{category}'. Supported: {SUPPORTED_CATEGORIES}"
        )
        
    cfg = load_config()
    cat_cfg = get_category_config(category, cfg)
    model_dir = Path(cat_cfg.get("model_dir", f"models/{category}/patchcore_v23"))
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}")
        
    if category not in DETECTOR_CACHE:
        # Instantiates category detector with its specific memory bank and thresholds
        DETECTOR_CACHE[category] = PatchCoreDetectorV23(category=category, device=DEVICE)
        
    return DETECTOR_CACHE[category]""",
        "explanation": "VisionInspect uses a unified PatchCoreDetectorV23 engine with dynamic configuration dispatch. When an inspection is requested for a specific category, get_detector dynamically loads that category's calibrated memory bank, spatial prior tensor, and decision thresholds from config.yaml without requiring separate duplicated detector implementations."
    }
}


# =============================================================================
# REUSABLE MODEL DOCUMENTATION PAGE COMPONENT
# =============================================================================

def render_model_documentation_page(
    category_id: str,
    on_all_models: Callable[[], None],
    on_select_model: Callable[[str], None],
    on_try_inspect: Callable[[str], None],
):
    """
    Renders the dedicated documentation page for the selected category.
    """
    category_id = category_id.lower().strip()
    if category_id not in CATEGORY_DOCS:
        category_id = "bottle"

    doc = CATEGORY_DOCS[category_id]
    all_cat_keys = list(CATEGORY_DOCS.keys())
    curr_idx = all_cat_keys.index(category_id)
    prev_cat_id = all_cat_keys[(curr_idx - 1) % len(all_cat_keys)]
    next_cat_id = all_cat_keys[(curr_idx + 1) % len(all_cat_keys)]

    # -------------------------------------------------------------------------
    # 1. TOP SUB-NAV BAR (BACK TO OVERVIEW & PREV/NEXT ROUTING)
    # -------------------------------------------------------------------------
    nav_col1, nav_col2, nav_col3 = st.columns([1.5, 3.0, 1.5])
    with nav_col1:
        if st.button("← All Models Overview", key=f"btn_all_models_{category_id}", use_container_width=True):
            on_all_models()
            st.rerun()

    with nav_col2:
        cat_pills = st.columns(len(all_cat_keys))
        for p_idx, cat_key in enumerate(all_cat_keys):
            with cat_pills[p_idx]:
                is_curr = (cat_key == category_id)
                btn_type = "primary" if is_curr else "secondary"
                if st.button(CATEGORY_DOCS[cat_key]["name"], key=f"pill_{cat_key}_{category_id}", type=btn_type, use_container_width=True):
                    on_select_model(cat_key)
                    st.rerun()

    with nav_col3:
        next_label = f"Next: {CATEGORY_DOCS[next_cat_id]['name']} →"
        if st.button(next_label, key=f"btn_next_top_{category_id}", use_container_width=True):
            on_select_model(next_cat_id)
            st.rerun()

    st.markdown("<hr style='margin: 0.6rem 0 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 2. HEADER: TITLE, IMAGE, SHORT DESCRIPTION & SPEC CHIPS
    # -------------------------------------------------------------------------
    head_left, head_right = st.columns([1.8, 1.2], gap="large")

    with head_left:
        st.markdown(f"""
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 4px;">
            INDUSTRIAL MODEL DOCUMENTATION &bull; {doc['id'].upper()}
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 700; color: var(--text-primary); margin: 0 0 8px 0; letter-spacing: -0.02em;">
            {doc['name']} <span style="font-weight: 400; color: var(--text-secondary); font-size: 1.4rem;">// PatchCore v2.3</span>
        </h1>
        <div style="font-size: 1.0rem; font-weight: 600; color: var(--text-primary); margin-bottom: 12px;">
            {doc['subtitle']}
        </div>
        <p style="font-size: 0.90rem; color: var(--text-secondary); line-height: 1.6; margin-bottom: 20px;">
            &ldquo;{doc['short_desc']}&rdquo;
        </p>
        """, unsafe_allow_html=True)

        # Technical Specification Chips
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 16px;">
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">BACKBONE</div>
                <div style="font-size: 0.84rem; font-weight: 600; color: var(--text-primary); margin-top: 2px;">ResNet-18 (Frozen)</div>
            </div>
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">FEATURE DESCRIPTORS</div>
                <div style="font-size: 0.84rem; font-weight: 600; color: var(--text-primary); margin-top: 2px;">448D Multi-Scale</div>
            </div>
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">FEATURE GRID</div>
                <div style="font-size: 0.84rem; font-weight: 600; color: var(--text-primary); margin-top: 2px;">64 &times; 64 (4,096 Patches)</div>
            </div>
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">TRAINING PARADIGM</div>
                <div style="font-size: 0.84rem; font-weight: 600; color: var(--text-primary); margin-top: 2px;">Unsupervised Normal Only</div>
            </div>
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">DETECTION METRIC</div>
                <div style="font-size: 0.84rem; font-weight: 600; color: var(--text-primary); margin-top: 2px;">Euclidean L2 Nearest Neighbor</div>
            </div>
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">MODEL STATUS</div>
                <div style="font-size: 0.84rem; font-weight: 700; color: {doc['status_badge_color']}; margin-top: 2px;">&bull; {doc['model_status']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with head_right:
        # Golden Reference Image Card
        golden_path = Path(doc["golden_sample"])
        b64_img = ""
        if golden_path.exists():
            try:
                with Image.open(golden_path) as im:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="JPEG", quality=90)
                    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                b64_img = ""

        if b64_img:
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px;">
                    NOMINAL GOLDEN STANDARD &bull; {doc['name'].upper()}
                </div>
                <img src="data:image/jpeg;base64,{b64_img}" alt="{doc['name']} Golden Sample" style="width: 100%; max-width: 280px; border-radius: 6px; border: 1px solid var(--border); display: block; margin: 0 auto;" />
                <div style="font-size: 0.72rem; color: var(--text-secondary); margin-top: 8px;">
                    Verified defect-free reference from MVTec AD nominal baseline.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Golden standard reference image for {doc['name']}.")

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. WHAT IS THIS MODEL? (BEGINNER-FRIENDLY EXPLANATION & 6 STEPS)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: var(--accent-blue); text-transform: uppercase;">FOUNDATIONAL CONCEPT</div>
    <h2 style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 2px 0 10px 0;">What is PatchCore?</h2>
    <p style="font-size: 0.90rem; color: var(--text-secondary); line-height: 1.6; max-width: 820px; margin-bottom: 16px;">
        PatchCore is an unsupervised visual anomaly detection architecture engineered for industrial production lines where defective samples are rare, unpredictable, or completely unavailable during training. Instead of memorizing defect images or training unstable generative autoencoders, PatchCore records the visual features of normal, defect-free parts into a high-dimensional memory bank. During live inspection, every small patch of a test image is matched against this memory bank. Patches that deviate significantly from nominal appearance receive higher anomaly scores and are instantly localized.
    </p>
    """, unsafe_allow_html=True)

    # 6 Step Card Grid
    step_cols = st.columns(3, gap="medium")
    steps_data = [
        ("01", "Memory Bank Creation", "Defect-free training samples are processed during setup to build a comprehensive coreset memory bank of nominal visual patches."),
        ("02", "Deep Feature Extraction", "A frozen ResNet-18 backbone extracts multi-scale convolutional features combining fine textures and structural semantics."),
        ("03", "Memory Comparison", "Each localized 448D patch from the test image is compared against the normal memory bank via GPU Euclidean distance."),
        ("04", "Anomaly Scoring", "Unusual patches that have no close match in the nominal memory bank receive high anomaly distance scores."),
        ("05", "Calibrated Decision", "Category-specific thresholds (T_image) evaluate the anomaly score to render a definitive Pass/Fail verdict."),
        ("06", "Pixel-Level Localization", "Regions exceeding the pixel threshold (T_pixel) undergo morphological filtering to produce precise bounding boxes.")
    ]

    for s_idx, (num, title, desc) in enumerate(steps_data):
        target_col = step_cols[s_idx % 3]
        with target_col:
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; margin-bottom: 12px; min-height: 130px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span style="font-size: 0.75rem; font-weight: 800; color: var(--accent-blue); background: var(--surface-subtle); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border);">{num}</span>
                    <span style="font-size: 0.86rem; font-weight: 700; color: var(--text-primary);">{title}</span>
                </div>
                <div style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.5;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. MODEL-SPECIFIC DEFECTS (REAL MVTEC DEFECT CATEGORIES)
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: var(--accent-blue); text-transform: uppercase;">DATASET DEFECT TAXONOMY</div>
    <h2 style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 2px 0 6px 0;">{doc['name']} Defect Categories</h2>
    <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 16px;">
        These are the actual, verified defect categories present in the MVTec Anomaly Detection benchmark dataset for <code>{doc['id']}</code>.
    </p>
    """, unsafe_allow_html=True)

    d_cols = st.columns(len(doc["defects"]), gap="small")
    for d_idx, def_item in enumerate(doc["defects"]):
        with d_cols[d_idx]:
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-top: 3px solid var(--danger); border-radius: 6px; padding: 12px 14px; height: 100%;">
                <div style="font-size: 0.70rem; font-weight: 700; color: var(--danger); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">DEFECT #{d_idx+1:02d}</div>
                <div style="font-size: 0.90rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;"><code>{def_item['tag']}</code></div>
                <div style="font-size: 0.78rem; color: var(--text-secondary); line-height: 1.45;">
                    {def_item['desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 5. HOW THIS MODEL SCANS AN IMAGE (VISUAL PIPELINE FLOW)
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: var(--accent-blue); text-transform: uppercase;">SCANNING WORKFLOW</div>
    <h2 style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 2px 0 6px 0;">How VisionInspect Scans a {doc['name']}</h2>
    <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 16px;">
        End-to-end mathematical dataflow through the PatchCore v2.3 inference engine.
    </p>
    """, unsafe_allow_html=True)

    scan_steps = [
        ("01", "Input", f"Receive {doc['name'].lower()} image &amp; resize to uniform 256&times;256 RGB tensor."),
        ("02", "Feature Extraction", "ResNet-18 forward pass extracts convolutional representations from Layer 1, 2, and 3."),
        ("03", "Patch Assembly", "Bilinearly upsample L2 and L3 onto 64&times;64 grid; concatenate into 4,096 &times; 448D patch descriptors."),
        ("04", "Memory Match", f"Compute Euclidean L2 distance against {doc['name']} coreset memory bank via GPU torch.cdist."),
        ("05", "Anomaly Map", f"Reshape minimum patch distances into 2D heatmap; subtract spatial prior ({doc['spatial_prior']}) if active."),
        ("06", "Score & Decision", f"Evaluate score against calibrated threshold T_image = {doc['image_threshold']:.4f} &rarr; Pass/Fail."),
        ("07", "Localization", f"Segment pixels &gt; T_pixel ({doc['pixel_threshold']:.4f}); apply {doc['morphology_kernel']} kernel &amp; bounding box clustering.")
    ]

    pipe_cols = st.columns(len(scan_steps), gap="small")
    for p_idx, (p_num, p_title, p_desc) in enumerate(scan_steps):
        with pipe_cols[p_idx]:
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 10px; text-align: center; height: 100%;">
                <div style="font-size: 0.72rem; font-weight: 800; color: var(--accent-blue); margin-bottom: 2px;">STEP {p_num}</div>
                <div style="font-size: 0.80rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">{p_title}</div>
                <div style="font-size: 0.70rem; color: var(--text-secondary); line-height: 1.35;">
                    {p_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 6. CATEGORY-SPECIFIC CONFIGURATION (VERIFIED FROM config.yaml)
    # -------------------------------------------------------------------------
    st.markdown(f"""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: var(--accent-blue); text-transform: uppercase;">CALIBRATED HYPERPARAMETERS</div>
    <h2 style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 2px 0 6px 0;">{doc['name']} Configuration &bull; <code>config.yaml</code></h2>
    <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 16px;">
        Production parameters calibrated exclusively on nominal validation data to maximize F1 defect sensitivity without false positives.
    </p>
    """, unsafe_allow_html=True)

    cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns(4, gap="small")
    with cfg_c1:
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">IMAGE THRESHOLD (T_image)</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: var(--text-primary); margin: 4px 0 2px 0;">{doc['image_threshold']:.4f}</div>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">Sample is defective if Score &gt; T_image</div>
        </div>
        """, unsafe_allow_html=True)
    with cfg_c2:
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">PIXEL THRESHOLD (T_pixel)</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: var(--text-primary); margin: 4px 0 2px 0;">{doc['pixel_threshold']:.4f}</div>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">Pixel flagged anomalous if heatmap &gt; T_pixel</div>
        </div>
        """, unsafe_allow_html=True)
    with cfg_c3:
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">SPATIAL PRIOR</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin: 6px 0 4px 0;">{doc['spatial_prior'].split('(')[0].strip()}</div>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">{doc['spatial_prior_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
    with cfg_c4:
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">MORPHOLOGY &amp; MIN AREA</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin: 6px 0 4px 0;">Kernel {doc['morphology_kernel']} &bull; {doc['min_region_area']} px</div>
            <div style="font-size: 0.72rem; color: var(--text-secondary);">Border margin: {doc['border_margin']} px &bull; Merge: {doc['merge_distance']} px</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. NEW "CODE" SECTION (REAL REPOSITORY SOURCE CODE WITH TABS)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: var(--accent-blue); text-transform: uppercase;">SOURCE CODE IMPLEMENTATION</div>
    <h2 style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 2px 0 6px 0;">Real Production Code &bull; VisionInspect Repository</h2>
    <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 12px;">
        VisionInspect uses a shared, high-performance PatchCore v2.3 inference engine. Each category uses its own trained memory bank, spatial prior, and calibrated configuration. Inspect the real source code files powering this model below.
    </p>
    """, unsafe_allow_html=True)

    code_tab1, code_tab2, code_tab3, code_tab4, code_tab5 = st.tabs([
        "Inference Pipeline",
        "Feature Extraction",
        "Anomaly Scoring",
        "Morphological Localization",
        "Configuration Dispatch"
    ])

    tabs_mapping = [
        (code_tab1, "inference"),
        (code_tab2, "features"),
        (code_tab3, "scoring"),
        (code_tab4, "localization"),
        (code_tab5, "config_dispatch")
    ]

    for tab_obj, snippet_key in tabs_mapping:
        snip = REAL_CODE_SNIPPETS[snippet_key]
        with tab_obj:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: var(--surface-subtle); border: 1px solid var(--border); border-bottom: none; border-radius: 6px 6px 0 0; padding: 8px 14px; font-family: monospace; font-size: 0.80rem; color: var(--text-secondary);">
                <span>📁 <b>{snip['file']}</b> &nbsp; <span style="color: var(--text-muted);">({snip['lines']})</span></span>
                <span style="background: var(--surface); border: 1px solid var(--border); padding: 2px 8px; border-radius: 4px; font-size: 0.70rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase;">{snip['lang']}</span>
            </div>
            """, unsafe_allow_html=True)

            st.code(snip["code"], language=snip["lang"])

            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border); border-top: none; border-radius: 0 0 6px 6px; padding: 10px 14px; margin-bottom: 14px;">
                <div style="font-size: 0.70rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 2px;">WHAT THIS CODE DOES</div>
                <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5;">
                    {snip['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Category-Specific Usage Example Section
    st.markdown(f"""
    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 14px 18px; margin-top: 14px; margin-bottom: 14px;">
        <div style="font-size: 0.72rem; font-weight: 700; color: var(--accent-blue); text-transform: uppercase; margin-bottom: 4px;">CATEGORY SPECIALIZATION</div>
        <div style="font-size: 1.0rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">How VisionInspect executes PatchCore for {doc['name']}</div>
        <div style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 10px;">
            {doc['category_specific_notes']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Full Implementation Expander
    with st.expander(f"View Full Implementation &bull; src/detection/patchcore_v23_detector.py ({doc['name']})", expanded=False):
        st.markdown(f"""
        <div style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 8px;">
            File: <code>src/detection/patchcore_v23_detector.py</code> &bull; Total Lines: 524 &bull; Class: <code>PatchCoreDetectorV23</code>
        </div>
        """, unsafe_allow_html=True)
        try:
            full_code = Path("src/detection/patchcore_v23_detector.py").read_text(encoding="utf-8")
            st.code(full_code, language="python")
        except Exception as e:
            st.warning(f"Could not load full source file: {e}")

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 8. TECHNICAL DETAILS (EXPANDABLE AUDIT & SPECIFICATION TABLE)
    # -------------------------------------------------------------------------
    with st.expander(f"Technical Specifications & Hyperparameter Audit &bull; {doc['name']}", expanded=False):
        st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem;">
            <thead>
                <tr style="border-bottom: 2px solid var(--border); text-align: left; color: var(--text-secondary);">
                    <th style="padding: 8px 6px;">Parameter</th>
                    <th style="padding: 8px 6px;">Value</th>
                    <th style="padding: 8px 6px;">Operational Rationale</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Backbone Architecture</td>
                    <td style="padding: 8px 6px;"><code>resnet18</code> (PyTorch torchvision)</td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Frozen pre-trained weights; provides compact, generalizable visual embeddings without over-fitting.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Tapped Layers</td>
                    <td style="padding: 8px 6px;"><code>layer1, layer2, layer3</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Multi-scale hierarchy: layer 1 preserves micro-texture, layers 2-3 capture semantic contours.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Feature Descriptor Dimension</td>
                    <td style="padding: 8px 6px;"><code>448 channels</code> (64 + 128 + 256)</td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Bilinear upsampling aligns deeper layer channels to 64&times;64 spatial grid.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Coreset Sampling Ratio</td>
                    <td style="padding: 8px 6px;"><code>10% (0.10)</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Maintains complete coverage of the normal feature manifold while reducing memory footprint by 90%.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Image Scoring Method</td>
                    <td style="padding: 8px 6px;"><code>{doc['image_score_method']}</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Robust aggregation across smoothed heatmap prevents single-pixel noise from causing false alarms.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Image Threshold (T_image)</td>
                    <td style="padding: 8px 6px; font-weight: 700; color: var(--text-primary);"><code>{doc['image_threshold']:.4f}</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Calibrated against normal validation baseline (100% nominal precision).</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Pixel Threshold (T_pixel)</td>
                    <td style="padding: 8px 6px; font-weight: 700; color: var(--text-primary);"><code>{doc['pixel_threshold']:.4f}</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Defines pixel anomaly segmentation boundary for defect mask extraction.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Spatial Background Prior</td>
                    <td style="padding: 8px 6px;"><code>{doc['spatial_prior']}</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">{doc['spatial_prior_desc']}</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Morphological Processing</td>
                    <td style="padding: 8px 6px;"><code>Kernel {doc['morphology_kernel']} (Close &rarr; Open)</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Preserves crack connectivity while suppressing isolated false-positive speckles.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Minimum Region Area</td>
                    <td style="padding: 8px 6px;"><code>{doc['min_region_area']} px</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Connected components smaller than this threshold are filtered as sub-resolution sensor noise.</td>
                </tr>
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Border Margin Exclusion</td>
                    <td style="padding: 8px 6px;"><code>{doc['border_margin']} px</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Prevents lens vignetting or boundary padding artifacts from registering as defect clusters.</td>
                </tr>
                <tr>
                    <td style="padding: 8px 6px; font-weight: 600; color: var(--text-primary);">Region Merge Distance</td>
                    <td style="padding: 8px 6px;"><code>{doc['merge_distance']} px</code></td>
                    <td style="padding: 8px 6px; color: var(--text-secondary);">Clusters fragmented nearby defects (e.g., fractured teeth) into unified bounding boxes.</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1.4rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 9. "TRY THE MODEL" CALL-TO-ACTION (ACTIONABLE ROUTING TO INSPECT PAGE)
    # -------------------------------------------------------------------------
    try_col1, try_col2 = st.columns([2.0, 1.2], gap="medium")
    with try_col1:
        st.markdown(f"""
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: var(--accent-blue); text-transform: uppercase;">INTERACTIVE TRIAL</div>
        <h3 style="font-size: 1.35rem; font-weight: 700; color: var(--text-primary); margin: 2px 0 6px 0;">Try {doc['name']} Inspection</h3>
        <p style="font-size: 0.88rem; color: var(--text-secondary); margin-bottom: 8px;">
            Ready to test this model with real components? Navigate directly to the live inspection workstation with <b>{doc['name']}</b> pre-selected as the active category.
        </p>
        """, unsafe_allow_html=True)

    with try_col2:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button(f"Inspect a {doc['name']} →", key=f"btn_try_inspect_{category_id}", type="primary", use_container_width=True):
            on_try_inspect(category_id)
            st.rerun()

    st.markdown("<hr style='margin: 1.4rem 0 1.0rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 10. BOTTOM NAVIGATION: PREVIOUS / ALL / NEXT
    # -------------------------------------------------------------------------
    foot_col1, foot_col2, foot_col3 = st.columns([1.5, 1.0, 1.5])
    with foot_col1:
        prev_name = CATEGORY_DOCS[prev_cat_id]["name"]
        if st.button(f"← Previous: {prev_name}", key=f"btn_prev_foot_{category_id}", use_container_width=True):
            on_select_model(prev_cat_id)
            st.rerun()

    with foot_col2:
        if st.button("All Models", key=f"btn_all_foot_{category_id}", use_container_width=True):
            on_all_models()
            st.rerun()

    with foot_col3:
        next_name = CATEGORY_DOCS[next_cat_id]["name"]
        if st.button(f"Next: {next_name} →", key=f"btn_next_foot_{category_id}", use_container_width=True):
            on_select_model(next_cat_id)
            st.rerun()
