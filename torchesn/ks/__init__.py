"""Kuramoto-Sivashinsky utilities for data generation."""

from .benchmark import benchmark_parallel_reservoir, rmse_over_space
from .ks_parallel_reservoir import KSParallelParams, KSParallelReservoir
from .reservoir_utils import (
    estimate_spectral_radius_power_iteration,
    generate_reservoir_sparse,
    make_block_input_weights,
)
from .ks_reservoir import KSBasicSingleReservoir, KSReservoirParams
from .plots import (
    reproduce_prl_fig4,
    reproduce_prl_fig5a,
    reproduce_prl_fig5b,
    reproduce_prl_fig6,
    reproduce_prl_figure2,
)
from .solver import generate_ks_dataset, ks_solve_etdrk4_forced, kursiv_solve_etdrk4

__all__ = [
    "benchmark_parallel_reservoir",
    "estimate_spectral_radius_power_iteration",
    "generate_reservoir_sparse",
    "generate_ks_dataset",
    "KSBasicSingleReservoir",
    "KSParallelParams",
    "KSParallelReservoir",
    "KSReservoirParams",
    "ks_solve_etdrk4_forced",
    "kursiv_solve_etdrk4",
    "make_block_input_weights",
    "reproduce_prl_fig4",
    "reproduce_prl_fig5a",
    "reproduce_prl_fig5b",
    "reproduce_prl_fig6",
    "reproduce_prl_figure2",
    "rmse_over_space",
]
