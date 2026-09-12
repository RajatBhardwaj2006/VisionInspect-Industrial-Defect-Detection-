import torch
import torch.nn.functional as F


def reconstruction_loss(original, reconstructed):
    """
    Calculate the mean squared reconstruction error.

    Higher loss = larger difference between the
    original image and reconstructed image.
    """

    return F.mse_loss(reconstructed, original)