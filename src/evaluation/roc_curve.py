"""
VisionInspect - ROC Curve and AUC Computation Utilities
"""
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute False Positive Rate, True Positive Rate, thresholds, and AUROC score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return fpr, tpr, thresholds, auc
