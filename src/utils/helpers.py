"""
VisionInspect - Utility Functions and Helper Routines
"""
import random
import torch
import numpy as np

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Return available torch device (cuda if GPU available, otherwise cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
