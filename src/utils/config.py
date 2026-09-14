import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path=None) -> Dict[str, Any]:
    """
    Load configuration from YAML file, with fallback defaults.
    """
    if config_path is None:
        # Default path is project root config.yaml
        config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)

    # Defaults
    defaults = {
        "model": {
            "image_size": 256
        },
        "detection": {
            "image_score_method": "max_raw",
            "image_threshold": 0.50,
            "pixel_threshold": 0.15
        },
        "localization": {
            "gaussian_sigma": 1.0,
            "morphology_kernel": 3,
            "min_region_area": 30,
            "max_region_area": 15000,
            "min_region_width": 9,
            "min_region_height": 9,
            "border_margin": 5,
            "merge_distance": 5
        }
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if not cfg:
                return defaults
            # Merge with defaults
            for section in defaults:
                if section in cfg:
                    for key in defaults[section]:
                        if key not in cfg[section]:
                            cfg[section][key] = defaults[section][key]
                else:
                    cfg[section] = defaults[section]
            return cfg
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}. Using defaults.")
        return defaults

def get_category_config(category: str, config: Optional[Dict[str, Any]] = None, config_path=None) -> Dict[str, Any]:
    """
    Retrieve category-specific configuration with fallbacks from patchcore_v23.
    """
    if config is None:
        config = load_config(config_path)
        
    categories_cfg = config.get("categories", {})
    base_v23 = config.get("patchcore_v23", config.get("patchcore_v22", {}))
    
    # Textures typically do not use fixed 2D spatial background priors
    known_textures = {"carpet", "grid", "leather", "tile", "wood"}
    default_spatial_prior = False if category in known_textures else True
    
    cat_specific = categories_cfg.get(category, {})
    # Fallback / override from phase3_2_categories if specified
    phase32_cat = config.get("phase3_2_categories", {}).get(category, {})
    for k, v in phase32_cat.items():
        if k not in cat_specific:
            cat_specific[k] = v
    
    merged = {
        "category": category,
        "model_dir": cat_specific.get("model_dir", f"models/{category}/patchcore_v23"),
        "use_spatial_prior": cat_specific.get("use_spatial_prior", default_spatial_prior),
        "prior_mode": cat_specific.get("prior_mode", "mean"),
        "image_score_method": cat_specific.get("image_score_method", config.get("detection", {}).get("image_score_method", "max_raw")),
        "image_threshold": cat_specific.get("image_threshold", base_v23.get("image_threshold", 1.50)),
        "pixel_threshold": cat_specific.get("pixel_threshold", base_v23.get("pixel_threshold", 1.40)),
        "gaussian_sigma": cat_specific.get("gaussian_sigma", base_v23.get("gaussian_sigma", 1.20)),
        "morphology_kernel": cat_specific.get("morphology_kernel", base_v23.get("morphology_kernel", 3)),
        "min_region_area": cat_specific.get("min_region_area", base_v23.get("min_region_area", 25)),
        "border_margin": cat_specific.get("border_margin", base_v23.get("border_margin", 5)),
        "merge_distance": cat_specific.get("merge_distance", base_v23.get("merge_distance", 15.0)),
        "coreset_sampling_ratio": cat_specific.get("coreset_sampling_ratio", base_v23.get("coreset_sampling_ratio", 0.10)),
        "backbone": cat_specific.get("backbone", base_v23.get("backbone", "resnet18")),
    }
    return merged
