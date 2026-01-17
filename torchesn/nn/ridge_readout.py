"""Ridge readout utilities for reservoir forecasting."""

from __future__ import annotations

import torch


def transform_state(states: torch.Tensor) -> torch.Tensor:
    """Square every other element of the state vector (MATLAB 2:2:N)."""
    transformed = states.clone()
    transformed[..., 1::2] = transformed[..., 1::2] ** 2
    return transformed


def ridge_regression(X: torch.Tensor, Y: torch.Tensor, beta: float) -> torch.Tensor:
    """Solve Wout = Y X^T (X X^T + beta I)^-1."""
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("X and Y must be 2D tensors")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must share the same time dimension")

    xx_t = X @ X.T
    if beta > 0:
        xx_t = xx_t + beta * torch.eye(xx_t.shape[0], device=xx_t.device, dtype=xx_t.dtype)

    identity = torch.eye(xx_t.shape[0], device=xx_t.device, dtype=xx_t.dtype)
    try:
        chol = torch.linalg.cholesky(xx_t)
        inv = torch.cholesky_solve(identity, chol)
    except RuntimeError:
        inv = torch.linalg.solve(xx_t, identity)

    return (Y @ X.T) @ inv
