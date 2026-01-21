"""Kuramoto-Sivashinsky utilities for data generation."""

from .reservoir_utils import (
    estimate_spectral_radius_power_iteration,
    generate_reservoir_sparse,
    make_block_input_weights,
)
from .ks_reservoir import KSBasicSingleReservoir, KSReservoirParams
from .plots import reproduce_prl_figure2
from .solver import kursiv_solve_etdrk4

__all__ = [
    "estimate_spectral_radius_power_iteration",
    "generate_reservoir_sparse",
    "KSBasicSingleReservoir",
    "KSReservoirParams",
    "kursiv_solve_etdrk4",
    "make_block_input_weights",
    "reproduce_prl_figure2",
]
