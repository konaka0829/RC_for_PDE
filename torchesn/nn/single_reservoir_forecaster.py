"""Single-reservoir forecaster with teacher forcing and autonomous prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from torchesn.nn.ridge_readout import ridge_regression, transform_state
from torchesn.nn.sparse_reservoir import SparseReservoir, build_sparse_reservoir


@dataclass
class ReservoirState:
    reservoir: SparseReservoir
    state: torch.Tensor
    Wout: Optional[torch.Tensor] = None


class SingleReservoirForecaster:
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        degree: int,
        spectral_radius: float,
        sigma: float = 1.0,
        beta: float = 1e-4,
        seed: int | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.beta = beta
        self.dtype = dtype
        self.device = device
        self.seed = seed

        self.reservoir = build_sparse_reservoir(
            reservoir_size=reservoir_size,
            num_inputs=input_dim,
            degree=degree,
            spectral_radius=spectral_radius,
            sigma=sigma,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        self.state = torch.zeros(self.reservoir.Win.shape[0], device=device, dtype=dtype)
        self.Wout: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        self.state = torch.zeros_like(self.state)

    def step(self, inputs: torch.Tensor) -> torch.Tensor:
        self.state = self.reservoir.step(self.state, inputs)
        return self.state

    def fit(self, inputs: torch.Tensor, washout: int = 0) -> torch.Tensor:
        if inputs.dim() != 2:
            raise ValueError("inputs must be a 2D tensor with shape (T, D)")
        if inputs.shape[1] != self.input_dim:
            raise ValueError("inputs dimension mismatch")
        if inputs.shape[0] < 2:
            raise ValueError("inputs must have at least 2 time steps")

        self.reset_state()
        states = []
        targets = []
        for t in range(inputs.shape[0] - 1):
            self.step(inputs[t])
            if t >= washout:
                states.append(transform_state(self.state).unsqueeze(1))
                targets.append(inputs[t + 1].unsqueeze(1))

        X = torch.cat(states, dim=1)
        Y = torch.cat(targets, dim=1)
        self.Wout = ridge_regression(X, Y, self.beta)
        return self.Wout

    def sync(self, inputs: torch.Tensor) -> None:
        if inputs.dim() != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError("inputs must be shape (T, D)")
        for t in range(inputs.shape[0]):
            self.step(inputs[t])

    def predict(self, initial_input: torch.Tensor, steps: int) -> torch.Tensor:
        if self.Wout is None:
            raise ValueError("Model has not been fit yet")
        if initial_input.shape != (self.input_dim,):
            raise ValueError("initial_input must have shape (D,)")

        outputs = torch.zeros((steps, self.input_dim), device=self.device, dtype=self.dtype)
        current = initial_input
        for t in range(steps):
            self.step(current)
            out = (self.Wout @ transform_state(self.state)).to(self.dtype)
            outputs[t] = out
            current = out
        return outputs
