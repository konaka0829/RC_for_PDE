"""Reservoir computing utilities matching MATLAB KSBasicSingleReservoir."""

from .ks_solver import ModelParams, kursiv_solve
from .reservoir import (
    ReservoirParams,
    augment_even_square,
    build_win,
    generate_reservoir_scipy,
    predict,
    reservoir_layer,
    scipy_sparse_to_torch,
    train_reservoir,
    train_reservoir_streaming,
    train_wout,
)

__all__ = [
    "ModelParams",
    "ReservoirParams",
    "augment_even_square",
    "build_win",
    "generate_reservoir_scipy",
    "kursiv_solve",
    "predict",
    "reservoir_layer",
    "scipy_sparse_to_torch",
    "train_reservoir",
    "train_reservoir_streaming",
    "train_wout",
]
