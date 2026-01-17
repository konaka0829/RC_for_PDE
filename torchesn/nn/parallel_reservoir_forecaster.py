"""Parallelized reservoir forecaster for spatially extended systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from torchesn.nn.ridge_readout import ridge_from_stats, transform_state
from torchesn.nn.sparse_reservoir import SparseReservoir, build_sparse_reservoir


@dataclass
class ParallelReservoirState:
    reservoir: SparseReservoir
    state: torch.Tensor
    Wout: Optional[torch.Tensor] = None


def build_group_indices(Q: int, g: int, l: int) -> List[torch.Tensor]:
    if Q % g != 0:
        raise ValueError("Q must be divisible by g")
    if l < 0:
        raise ValueError("l must be non-negative")
    q = Q // g

    indices = []
    for i in range(g):
        center_start = i * q
        center_indices = torch.arange(center_start, center_start + q)
        left = torch.arange(center_start - l, center_start)
        right = torch.arange(center_start + q, center_start + q + l)
        combined = torch.cat([left, center_indices, right]) % Q
        indices.append(combined)
    return indices


def rmse(u_true: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((u_true - u_pred) ** 2, dim=-1))


def mean_rmse_over_segments(rmse_series: torch.Tensor, segments: int) -> torch.Tensor:
    if segments <= 0:
        raise ValueError("segments must be positive")
    if rmse_series.numel() < segments:
        raise ValueError("rmse_series too short for requested segments")

    segment_length = rmse_series.numel() // segments
    trimmed = rmse_series[: segment_length * segments]
    reshaped = trimmed.view(segments, segment_length)
    return reshaped.mean(dim=1)


class ParallelReservoirForecaster:
    def __init__(
        self,
        Q: int,
        g: int,
        l: int,
        reservoir_size_approx: int,
        degree: int,
        spectral_radius: float,
        sigma: float = 1.0,
        beta: float = 1e-4,
        seed: int | None = None,
        share_weights: bool = False,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if Q % g != 0:
            raise ValueError("Q must be divisible by g")
        if g <= 0:
            raise ValueError("g must be positive")
        if l < 0:
            raise ValueError("l must be non-negative")

        self.Q = Q
        self.g = g
        self.l = l
        self.q = Q // g
        self.input_dim = self.q + 2 * l
        self.reservoir_size_approx = reservoir_size_approx
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.sigma = sigma
        self.beta = beta
        self.seed = seed
        self.share_weights = share_weights
        self.device = device
        self.dtype = dtype

        self.input_indices = build_group_indices(Q=Q, g=g, l=l)
        self.center_indices = [torch.arange(i * self.q, (i + 1) * self.q) for i in range(g)]

        self.states: List[ParallelReservoirState] = []
        self._build_reservoirs()

    def _build_reservoirs(self) -> None:
        if self.share_weights:
            shared = build_sparse_reservoir(
                reservoir_size=self.reservoir_size_approx,
                num_inputs=self.input_dim,
                degree=self.degree,
                spectral_radius=self.spectral_radius,
                sigma=self.sigma,
                seed=self.seed,
                dtype=self.dtype,
                device=self.device,
            )
            state = torch.zeros(shared.Win.shape[0], device=self.device, dtype=self.dtype)
            self.states = [
                ParallelReservoirState(shared, state.clone()) for _ in range(self.g)
            ]
            return

        self.states = []
        for i in range(self.g):
            reservoir = build_sparse_reservoir(
                reservoir_size=self.reservoir_size_approx,
                num_inputs=self.input_dim,
                degree=self.degree,
                spectral_radius=self.spectral_radius,
                sigma=self.sigma,
                seed=None if self.seed is None else self.seed + i,
                dtype=self.dtype,
                device=self.device,
            )
            state = torch.zeros(reservoir.Win.shape[0], device=self.device, dtype=self.dtype)
            self.states.append(ParallelReservoirState(reservoir, state))

    def reset_states(self) -> None:
        for entry in self.states:
            entry.state = torch.zeros_like(entry.state)

    def _fit_single(
        self,
        entry: ParallelReservoirState,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        washout: int,
        chunk_size: int,
    ) -> torch.Tensor:
        if inputs.shape[0] < 2:
            raise ValueError("inputs must have at least 2 time steps")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        entry.state = torch.zeros_like(entry.state)
        Dr = entry.state.shape[0]
        Dout = targets.shape[1]
        XXT = torch.zeros((Dr, Dr), device=inputs.device, dtype=inputs.dtype)
        YXT = torch.zeros((Dout, Dr), device=inputs.device, dtype=inputs.dtype)
        x_chunk: List[torch.Tensor] = []
        y_chunk: List[torch.Tensor] = []
        sample_count = 0

        with torch.no_grad():
            for t in range(inputs.shape[0] - 1):
                entry.state = entry.reservoir.step(entry.state, inputs[t])
                if t >= washout:
                    x_chunk.append(transform_state(entry.state))
                    y_chunk.append(targets[t + 1])
                    sample_count += 1
                    if len(x_chunk) >= chunk_size:
                        X = torch.stack(x_chunk, dim=1)
                        Y = torch.stack(y_chunk, dim=1)
                        XXT = XXT + X @ X.T
                        YXT = YXT + Y @ X.T
                        x_chunk.clear()
                        y_chunk.clear()

            if x_chunk:
                X = torch.stack(x_chunk, dim=1)
                Y = torch.stack(y_chunk, dim=1)
                XXT = XXT + X @ X.T
                YXT = YXT + Y @ X.T

        if sample_count == 0:
            raise ValueError("washout leaves no samples for training")

        return ridge_from_stats(XXT, YXT, self.beta)

    def fit(self, train_u: torch.Tensor, washout: int = 0, chunk_size: int = 1024) -> None:
        if train_u.dim() != 2 or train_u.shape[1] != self.Q:
            raise ValueError("train_u must have shape (T, Q)")

        train_u = train_u.to(device=self.device, dtype=self.dtype)

        if self.share_weights:
            inputs = train_u[:, self.input_indices[0]]
            targets = train_u[:, self.center_indices[0]]
            Wout = self._fit_single(self.states[0], inputs, targets, washout, chunk_size)
            for entry in self.states:
                entry.Wout = Wout.clone()
            return

        for i, entry in enumerate(self.states):
            inputs = train_u[:, self.input_indices[i]]
            targets = train_u[:, self.center_indices[i]]
            entry.Wout = self._fit_single(entry, inputs, targets, washout, chunk_size)

    def synchronize(self, sync_inputs: torch.Tensor) -> None:
        if sync_inputs.dim() != 2 or sync_inputs.shape[1] != self.Q:
            raise ValueError("sync_inputs must have shape (T, Q)")
        sync_inputs = sync_inputs.to(device=self.device, dtype=self.dtype)

        self.reset_states()
        for t in range(sync_inputs.shape[0]):
            for i, entry in enumerate(self.states):
                entry.state = entry.reservoir.step(entry.state, sync_inputs[t, self.input_indices[i]])

    def predict(self, sync_inputs: torch.Tensor, predict_length: int) -> torch.Tensor:
        if predict_length <= 0:
            raise ValueError("predict_length must be positive")
        if any(entry.Wout is None for entry in self.states):
            raise ValueError("Model has not been fit yet")

        self.synchronize(sync_inputs)

        pred = torch.zeros((predict_length, self.Q), device=self.device, dtype=self.dtype)
        for t in range(predict_length):
            group_outputs: Sequence[torch.Tensor] = []
            for entry in self.states:
                out = entry.Wout @ transform_state(entry.state)
                group_outputs.append(out)

            u_hat = torch.cat(list(group_outputs), dim=0)
            pred[t] = u_hat

            for i, entry in enumerate(self.states):
                feedback = u_hat[self.input_indices[i]]
                entry.state = entry.reservoir.step(entry.state, feedback)

        return pred
