import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def compute_max_anomaly_score(anomaly_map):
    """
    Instead of averaging the whole anomaly map into one number
    (which is what caused defective images to look 'normal'),
    take the MAX value in the map.

    Why: a defect is usually a small region. Averaging it with
    hundreds of normal pixels drowns out the signal. The max
    value captures "the single worst spot" instead.

    Args:
        anomaly_map: numpy array or tensor, shape [H, W]

    Returns:
        A single float score
    """
    return float(np.max(anomaly_map))


def compute_image_level_auroc(scores, labels):
    """
    Computes AUROC: how well our scores separate good (0) vs
    defective (1) images, across ALL test images at once.

    An AUROC of 0.5 = random guessing.
    An AUROC of 1.0 = perfect separation.

    Args:
        scores: list of anomaly scores (one per image)
        labels: list of 0 (good) or 1 (defective), same order as scores

    Returns:
        AUROC score (float between 0 and 1)
    """
    return roc_auc_score(labels, scores)

def find_best_threshold(scores, labels):
    """
    Finds the anomaly score threshold that best separates
    good (0) from defective (1) images, using Youden's J
    statistic (maximizes true-positive rate minus false-positive rate).

    Args:
        scores: list of anomaly scores
        labels: list of 0/1 labels, same order as scores

    Returns:
        best_threshold: float
        best_tpr: true positive rate at that threshold
        best_fpr: false positive rate at that threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()

    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]

def compute_topk_anomaly_score(anomaly_map, k_percent=1.0):
    """
    Averages the top K% most anomalous pixels, instead of just
    the single max pixel. Balances sensitivity to both sharp
    small defects AND diffuse spread-out defects like contamination.

    Args:
        anomaly_map: numpy array, shape [H, W]
        k_percent: percentage of top pixels to average (default 1%)

    Returns:
        Float score
    """
    flat = anomaly_map.flatten()
    k = max(1, int(len(flat) * (k_percent / 100)))
    top_k_values = np.partition(flat, -k)[-k:]
    return float(np.mean(top_k_values))