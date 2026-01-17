"""Sparse reservoir utilities for single-reservoir forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class SparseReservoir:
    A: torch.Tensor
    Win: torch.Tensor

    def step(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        pre = torch.sparse.mm(self.A, state.unsqueeze(1)).squeeze(1) + self.Win @ inputs
        return torch.tanh(pre)


def generate_sparse_reservoir(
    size: int,
    degree: int,
    spectral_radius: float,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    power_iters: int = 100,
) -> torch.Tensor:
    """Generate a sparse reservoir adjacency matrix with target spectral radius."""
    if degree <= 0:
        raise ValueError("degree must be positive")

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    num_edges = size * degree
    rows = torch.randint(0, size, (num_edges,), generator=generator, device=device)
    cols = torch.randint(0, size, (num_edges,), generator=generator, device=device)
    values = (2.0 * torch.rand(num_edges, generator=generator, device=device) - 1.0).to(dtype)

    indices = torch.stack([rows, cols])
    A = torch.sparse_coo_tensor(indices, values, (size, size), device=device, dtype=dtype)
    A = A.coalesce()

    radius = _estimate_spectral_radius(A, power_iters=power_iters)
    if radius == 0.0:
        raise ValueError("estimated spectral radius is zero; check degree")

    scale = spectral_radius / radius
    scaled_values = A.values() * scale
    return torch.sparse_coo_tensor(A.indices(), scaled_values, A.shape, device=device, dtype=dtype).coalesce()


def _estimate_spectral_radius(A: torch.Tensor, power_iters: int = 100) -> float:
    size = A.shape[0]
    device = A.device
    dtype = A.dtype

    vec = torch.randn(size, device=device, dtype=dtype)
    vec = vec / (vec.norm() + 1e-12)

    radius = 0.0
    for _ in range(power_iters):
        vec = torch.sparse.mm(A, vec.unsqueeze(1)).squeeze(1)
        norm = vec.norm()
        if norm == 0:
            return 0.0
        vec = vec / norm
        radius = norm.item()

    return float(radius)


def generate_input_matrix(
    reservoir_size: int,
    num_inputs: int,
    sigma: float,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, int, int]:
    """Generate block-structured input weights and adjusted reservoir size."""
    if num_inputs <= 0:
        raise ValueError("num_inputs must be positive")

    nodes_per_input = int(round(reservoir_size / num_inputs))
    if nodes_per_input <= 0:
        raise ValueError("reservoir_size too small for num_inputs")

    size = nodes_per_input * num_inputs

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed + 1)

    Win = torch.zeros((size, num_inputs), device=device, dtype=dtype)
    for i in range(num_inputs):
        start = i * nodes_per_input
        end = (i + 1) * nodes_per_input
        Win[start:end, i] = (2.0 * torch.rand(nodes_per_input, generator=generator, device=device) - 1.0) * sigma

    return Win, size, nodes_per_input


def build_sparse_reservoir(
    reservoir_size: int,
    num_inputs: int,
    degree: int,
    spectral_radius: float,
    sigma: float,
    seed: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SparseReservoir:
    Win, size, _ = generate_input_matrix(
        reservoir_size=reservoir_size,
        num_inputs=num_inputs,
        sigma=sigma,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    A = generate_sparse_reservoir(
        size=size,
        degree=degree,
        spectral_radius=spectral_radius,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    return SparseReservoir(A=A, Win=Win)
