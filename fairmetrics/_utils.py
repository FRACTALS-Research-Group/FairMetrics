"""
    Utility functions for the FairMetrics package.
"""
import torch


def combine_features(features):
    """
    Combine the separated features of each entry into a single custom feature matrix.
    Parameters:
    - features: torch.Tensor
        Input feature matrix, either 1D or 2D.

    Returns:
    - torch.Tensor
        Combined feature matrix with shape (num_of_entries X num_of_features).
    """
    if features.dim() == 1:
        features = torch.unsqueeze(features, 1)
    else:
        features = features.T
    return features


__all__ = ["combine_features"]