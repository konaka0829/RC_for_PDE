from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import torch

from .ks_parallel_reservoir import KSParallelParams, KSParallelReservoir
from .solver import _as_device


@torch.no_grad()
def rmse_over_space(true_u: torch.Tensor, pred_u: torch.Tensor) -> torch.Tensor:
    """Compute RMSE across the spatial dimension at each time step.

    Args:
        true_u: [Q,T]
        pred_u: [Q,T]

    Returns:
        rmse: [T]
    """
    if true_u.shape != pred_u.shape:
        raise ValueError("Shapes must match")
    return torch.sqrt(torch.mean((true_u - pred_u) ** 2, dim=0))


def default_nonoverlapping_markers(
    *,
    discard_length: int,
    train_length: int,
    epsilon: int,
    tau: int,
    K: int,
) -> List[int]:
    """Default marker indices matching the paper's K nonoverlapping intervals."""
    if any(x < 0 for x in (discard_length, train_length, epsilon, tau, K)):
        raise ValueError("All lengths must be non-negative")
    if tau <= 0 or K <= 0:
        raise ValueError("tau and K must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")

    need = int(discard_length + train_length)
    m0 = int(need - epsilon)
    if m0 < 0:
        raise ValueError("discard_length+train_length must be >= epsilon")
    return [m0 + k * int(tau) for k in range(int(K))]


@torch.no_grad()
def benchmark_parallel_reservoir(
    u: torch.Tensor,
    *,
    params: KSParallelParams,
    dt: float = 0.25,
    lambda_max: float = 0.09,
    tau: int = 1000,
    K: int = 30,
    epsilon: int = 10,
    num_trials: int = 10,
    identical_reservoirs: bool = False,
    markers: Optional[Sequence[int]] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    time_chunk: int = 256,
) -> Dict[str, torch.Tensor]:
    """Run the PRL-style evaluation loop and return RMSE curves."""
    dev = _as_device(device)
    U = u
    if U.ndim != 2:
        raise ValueError("u must be a 2D tensor")
    if U.shape[0] != params.Q and U.shape[1] != params.Q:
        raise ValueError(f"u must have Q={params.Q} along one dimension")
    if U.shape[0] == params.Q:
        U = U.contiguous()
    else:
        U = U.T.contiguous()
    U = U.to(device=dev, dtype=dtype)

    need = int(params.discard_length + params.train_length)
    if U.shape[1] < need:
        raise ValueError("u is shorter than discard_length+train_length")

    if markers is None:
        markers = default_nonoverlapping_markers(
            discard_length=int(params.discard_length),
            train_length=int(params.train_length),
            epsilon=int(epsilon),
            tau=int(tau),
            K=int(K),
        )

    max_req = int(max(markers) + int(epsilon) + int(tau))
    if U.shape[1] < max_req:
        raise ValueError(
            f"u is too short for the requested markers: need at least {max_req} time steps, got {U.shape[1]}"
        )

    t_axis = torch.arange(0, int(tau), device=dev, dtype=dtype) * float(dt) * float(lambda_max)

    rmse_trials = torch.empty((int(num_trials), int(tau)), device=dev, dtype=dtype)

    for trial in range(int(num_trials)):
        trial_params = replace(params, jobid=int(params.jobid) + trial, predict_length=int(tau))
        model = KSParallelReservoir(trial_params, device=dev, dtype=dtype)
        model.fit(U[:, :need], identical_reservoirs=bool(identical_reservoirs), time_chunk=int(time_chunk))

        interval_rmses: List[torch.Tensor] = []
        for m in markers[: int(K)]:
            pred = model.predict_one_interval(U, warmup_start=int(m), sync_length=int(epsilon), predict_length=int(tau))
            true = float(trial_params.sigma) * U[:, int(m) + int(epsilon) : int(m) + int(epsilon) + int(tau)]
            interval_rmses.append(rmse_over_space(true, pred))

        rmse_trials[trial] = torch.stack(interval_rmses, dim=0).mean(dim=0)

    rmse_mean = rmse_trials.mean(dim=0)
    rmse_std = rmse_trials.std(dim=0, unbiased=False)

    return {
        "t": t_axis,
        "rmse_trials": rmse_trials,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
    }
