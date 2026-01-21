from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .reservoir_utils import generate_reservoir_sparse, make_block_input_weights
from .solver import _as_device


@dataclass
class KSReservoirParams:
    """Parameter bundle matching the MATLAB scripts."""

    # Reservoir
    radius: float = 0.6
    degree: float = 3.0
    sigma: float = 1.0
    beta: float = 1e-4
    reservoir_size: int = 5000

    # Training/prediction lengths
    train_length: int = 70_000
    predict_length: int = 1_000


class KSBasicSingleReservoir(torch.nn.Module):
    """Single-reservoir ESN matching the MATLAB KSBasicSingleReservoir implementation."""

    def __init__(
        self,
        *,
        num_inputs: int,
        reservoir_size: int,
        radius: float = 0.6,
        degree: float = 3.0,
        sigma: float = 1.0,
        beta: float = 1e-4,
        seed_A: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
        power_iter: int = 100,
    ) -> None:
        super().__init__()
        if num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive")
        if reservoir_size % num_inputs != 0:
            raise ValueError("reservoir_size must be divisible by num_inputs")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.num_inputs = int(num_inputs)
        self.N = int(reservoir_size)
        self.q = self.N // self.num_inputs
        self.radius = float(radius)
        self.degree = float(degree)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.dtype = dtype
        self.device = _as_device(device)

        A = generate_reservoir_sparse(
            self.N,
            self.radius,
            self.degree,
            seed=seed_A,
            device=self.device,
            dtype=self.dtype,
            power_iter=power_iter,
        )
        w_in_diag = make_block_input_weights(
            self.N,
            self.num_inputs,
            self.sigma,
            device=self.device,
            dtype=self.dtype,
        )

        self.register_buffer("A", A)
        self.register_buffer("w_in_diag", w_in_diag)
        self.register_buffer("w_out", torch.zeros(self.num_inputs, self.N, device=self.device, dtype=self.dtype))
        self.register_buffer("state", torch.zeros(self.N, device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.state.zero_()

    @torch.no_grad()
    def _augment_state_inplace(self, X: torch.Tensor) -> torch.Tensor:
        """Apply MATLAB's even-index squaring feature map."""
        X[1::2].square_()
        return X

    @torch.no_grad()
    def _input_term(self, u: torch.Tensor) -> torch.Tensor:
        """Compute win*u efficiently for the block-structured Win."""
        if u.ndim != 1 or u.shape[0] != self.num_inputs:
            raise ValueError(f"u must have shape [{self.num_inputs}]")
        u_rep = u.repeat_interleave(self.q)
        return self.w_in_diag * u_rep

    @torch.no_grad()
    def step(self, u: torch.Tensor) -> torch.Tensor:
        """Teacher-forced reservoir update: x <- tanh(Ax + Win*u)."""
        Ax = torch.sparse.mm(self.A, self.state.unsqueeze(1)).squeeze(1)
        self.state = torch.tanh(Ax + self._input_term(u))
        return self.state

    @torch.no_grad()
    def fit(self, u_train: torch.Tensor, *, chunk_size: int = 2048) -> "KSBasicSingleReservoir":
        """Train Wout by ridge regression (closed form), matching MATLAB train.m."""
        if u_train.ndim != 2 or u_train.shape[0] != self.num_inputs:
            raise ValueError(f"u_train must have shape [{self.num_inputs}, T]")
        T = int(u_train.shape[1])
        if T < 2:
            raise ValueError("Need at least 2 time steps for training")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        u_train = u_train.to(device=self.device, dtype=self.dtype)
        self.reset_state()

        S = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)
        YX = torch.zeros((self.num_inputs, self.N), device=self.device, dtype=self.dtype)

        t = 0
        x = self.state

        while t < T:
            end = min(T, t + int(chunk_size))
            L = end - t

            X = torch.empty((self.N, L), device=self.device, dtype=self.dtype)

            for j in range(L):
                X[:, j] = x
                tj = t + j
                if tj < T - 1:
                    Ax = torch.sparse.mm(self.A, x.unsqueeze(1)).squeeze(1)
                    x = torch.tanh(Ax + self._input_term(u_train[:, tj]))

            Phi = self._augment_state_inplace(X)

            Y_chunk = u_train[:, t:end]
            S.addmm_(Phi, Phi.T, beta=1.0, alpha=1.0)
            YX.addmm_(Y_chunk, Phi.T, beta=1.0, alpha=1.0)

            t = end

        self.state = x

        if self.beta > 0:
            S = S + (self.beta * torch.eye(self.N, device=self.device, dtype=self.dtype))

        Lchol = torch.linalg.cholesky(S)
        w_out_T = torch.cholesky_solve(YX.T, Lchol)
        self.w_out = w_out_T.T
        return self

    @torch.no_grad()
    def predict(self, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autonomous prediction (closed-loop), matching MATLAB predict.m."""
        if steps <= 0:
            raise ValueError("steps must be positive")

        outputs = torch.empty((self.num_inputs, int(steps)), device=self.device, dtype=self.dtype)
        x = self.state

        for i in range(int(steps)):
            x_aug = x.clone()
            self._augment_state_inplace(x_aug)
            out = self.w_out @ x_aug
            outputs[:, i] = out
            Ax = torch.sparse.mm(self.A, x.unsqueeze(1)).squeeze(1)
            x = torch.tanh(Ax + self._input_term(out))

        self.state = x
        return outputs, x
