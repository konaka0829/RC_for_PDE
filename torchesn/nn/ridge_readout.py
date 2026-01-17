"""Ridge readout utilities for reservoir forecasting."""

from __future__ import annotations

import torch


def transform_state(states: torch.Tensor) -> torch.Tensor:
    """Square every other element of the state vector (MATLAB 2:2:N)."""
    transformed = states.clone()
    transformed[..., 1::2] = transformed[..., 1::2] ** 2
    return transformed


def ridge_from_stats(XXT: torch.Tensor, YXT: torch.Tensor, beta: float) -> torch.Tensor:
    """Solve ridge regression from sufficient statistics."""
    if XXT.dim() != 2 or YXT.dim() != 2:
        raise ValueError("XXT and YXT must be 2D tensors")
    if XXT.shape[0] != XXT.shape[1]:
        raise ValueError("XXT must be square")
    if YXT.shape[1] != XXT.shape[0]:
        raise ValueError("YXT must have shape (Dout, Dr)")

    A = XXT
    if beta > 0:
        A = A + beta * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    rhs = YXT.T
    try:
        chol = torch.linalg.cholesky(A)
        sol = torch.cholesky_solve(rhs, chol)
    except RuntimeError:
        sol = torch.linalg.solve(A, rhs)

    return sol.T


def ridge_regression(X: torch.Tensor, Y: torch.Tensor, beta: float) -> torch.Tensor:
    """Solve Wout = Y X^T (X X^T + beta I)^-1."""
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("X and Y must be 2D tensors")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must share the same time dimension")

    XXT = X @ X.T
    YXT = Y @ X.T
    return ridge_from_stats(XXT, YXT, beta)
