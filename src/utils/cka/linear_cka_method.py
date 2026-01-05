import numpy as np
import torch


def linear_cka_np(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = (K * L).sum()
    return float(hsic / np.sqrt((K * K).sum() * (L * L).sum() + 1e-8))


def linear_cka_torch(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute Linear CKA similarity between two activation matrices."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    hsic = (K * L).sum()
    norm_x = (K * K).sum().sqrt()
    norm_y = (L * L).sum().sqrt()

    return float(hsic / (norm_x * norm_y + 1e-8))
