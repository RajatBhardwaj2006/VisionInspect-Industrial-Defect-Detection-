"""
VisionInspect - Evaluation Visualization Utilities
"""
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_inspection_result(
    image: np.ndarray,
    heatmap: np.ndarray,
    segmented_mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot original inspection image, anomaly heatmap, and binary defect mask."""
    cols = 3 if segmented_mask is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Anomaly Heatmap")
    axes[1].axis("off")
    
    if segmented_mask is not None:
        axes[2].imshow(segmented_mask, cmap="gray")
        axes[2].set_title("Defect Segmentation")
        axes[2].axis("off")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
