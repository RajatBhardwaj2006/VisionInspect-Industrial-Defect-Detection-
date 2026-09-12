import torch


def compute_anomaly_map(original, reconstructed):
    """
    Compute a pixel-level anomaly map.

    Larger values indicate a larger difference between
    the original and reconstructed image.
    """

    difference = torch.abs(original - reconstructed)

    # Average the RGB channels
    anomaly_map = difference.mean(dim=1)

    return anomaly_map