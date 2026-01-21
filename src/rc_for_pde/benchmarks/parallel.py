"""Benchmarking utilities to reproduce PRL 120, 024102 (2018) Figs. 4/5/6.

This is a PyTorch port of the *evaluation loop* described in the paper and
implemented (for pre-generated datasets) in MATLAB as:
    KSParallelReservoir/parallel_reservoir_benchmarking.m

What this module provides
-------------------------
1) A forced Kuramoto–Sivashinsky (KS) solver
       y_t = -y y_x - y_xx - y_xxxx + mu * cos(2*pi*x/lambda)
   using the ETDRK4 scheme of Kassam & Trefethen (2005).

2) A benchmarking loop that:
   - trains a KSParallelReservoir once
   - performs K prediction intervals, each of length tau
   - uses epsilon teacher-forced steps to synchronize before each interval
   - computes RMSE(t) averaged over K intervals
   - repeats this for multiple random reservoir realizations (trials)
     and averages RMSE(t) over trials

3) Convenience functions that create the plots for PRL Figs. 4/5/6.

Notes
-----
- The original MATLAB code distributes the g reservoirs across workers.
  This Python port runs in a single process and simulates the synchronous
  neighbour exchange.
- Reproducing the paper's largest system sizes (e.g. L=1600, Q=4096, g=512)
  is computationally heavy. The code is written to be faithful and clear;
  you may want to run with fewer trials/intervals first.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from rc_for_pde.ks.solver import generate_ks_dataset, ks_solve_etdrk4_forced
from rc_for_pde.reservoirs.parallel import KSParallelParams, KSParallelReservoir, rmse_over_space


def _as_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)




def default_nonoverlapping_markers(
    *,
    discard_length: int,
    train_length: int,
    epsilon: int,
    tau: int,
    K: int,
) -> List[int]:
    """Default marker indices matching the paper's K nonoverlapping intervals.

    Training uses data indices [0, ..., discard_length+train_length-1].
    The *first* prediction interval is taken to start right after training.

    We set the warm-up start (MATLAB: prediction_marker) to:
        m0 = (discard_length + train_length) - epsilon
    so that after epsilon teacher-forced steps, prediction begins at
        t0 = discard_length + train_length.

    Subsequent prediction intervals start every tau steps.
    """
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
    """Run the PRL-style evaluation loop and return RMSE curves.

    Args:
        u: unscaled KS data, shape [Q, T] (or [T, Q]).
        params: KSParallelParams for the reservoir.
        dt: KS sampling time step (for the returned time axis).
        lambda_max: largest Lyapunov exponent used to scale time.
        tau: prediction length per interval (paper: 1000).
        K: number of nonoverlapping intervals (paper: 30).
        epsilon: warm-up (synchronization) steps before each interval (paper: 10).
        num_trials: number of random reservoir realizations (paper: 10).
        identical_reservoirs: if True, train one reservoir and copy weights to all.
        markers: optional explicit warm-up start indices. If None, uses
            `default_nonoverlapping_markers`.
        device/dtype: torch placement.
        time_chunk: training accumulation chunk size.

    Returns:
        dict with keys:
            - "t": [tau] time axis in Lyapunov time units (lambda_max * t)
            - "rmse_trials": [num_trials, tau]
            - "rmse_mean": [tau]
            - "rmse_std": [tau]
    """
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

    # Time axis: treat the first predicted sample as t=0.
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


# ---------------------------- Figure helpers ----------------------------


@torch.no_grad()
def reproduce_prl_fig4(
    *,
    seed_data: int = 0,
    seed_trial: int = 0,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    # KS parameters (Fig.4)
    L: float = 200.0,
    Q: int = 512,
    mu: float = 0.01,
    wavelength: float = 100.0,
    dt: float = 0.25,
    lambda_max: float = 0.09,
    # Reservoir parameters (paper defaults)
    g: int = 64,
    locality: int = 6,
    approx_reservoir_size: int = 5000,
    radius: float = 0.6,
    degree: float = 3.0,
    beta: float = 1e-4,
    sigma: float = 1.0,
    discard_length: int = 1000,
    train_length: int = 70_000,
    epsilon: int = 10,
    tau: int = 1000,
):
    """Create a Fig.4-like 4-panel plot.

    Panels:
      (a) true KS field
      (b) reservoir prediction
      (c) true - prediction
      (d) true - KS-integration-from-predicted-initial-condition

    Returns:
        fig, dict with tensors for the panels (all [Q, tau]).
    """
    import matplotlib.pyplot as plt

    dev = _as_device(device)
    need = int(discard_length + train_length)
    total_steps = int(need + tau)

    u = generate_ks_dataset(
        Q=Q,
        L=L,
        mu=mu,
        wavelength=wavelength,
        dt=dt,
        n_steps=total_steps,
        burn_in=0,
        seed=seed_data,
        device=dev,
        dtype=dtype,
    )

    params = KSParallelParams(
        Q=Q,
        g=g,
        locality=locality,
        approx_reservoir_size=approx_reservoir_size,
        radius=radius,
        degree=degree,
        beta=beta,
        sigma=sigma,
        discard_length=discard_length,
        train_length=train_length,
        predict_length=tau,
        jobid=1 + int(seed_trial),
    )

    model = KSParallelReservoir(params, device=dev, dtype=dtype)
    model.fit(u[:, :need], identical_reservoirs=False)

    marker = need - int(epsilon)
    pred = model.predict_one_interval(u, warmup_start=int(marker), sync_length=int(epsilon), predict_length=int(tau))
    true = float(sigma) * u[:, int(marker) + int(epsilon) : int(marker) + int(epsilon) + int(tau)]
    err_res = true - pred

    # Baseline: integrate the KS equation from the *predicted* initial condition.
    init_raw = (pred[:, 0] / float(sigma)).to(device=dev, dtype=dtype)
    integ = torch.empty((int(tau), Q), device=dev, dtype=dtype)
    integ[0] = init_raw
    if tau > 1:
        integ[1:] = ks_solve_etdrk4_forced(
            init_raw,
            dt=float(dt),
            n_steps=int(tau - 1),
            L=float(L),
            mu=float(mu),
            wavelength=float(wavelength),
            device=dev,
            dtype=dtype,
        )
    integ_scaled = float(sigma) * integ.T  # [Q, tau]
    err_int = true - integ_scaled

    # Plot axes: x=Lyapunov time, y=space index (or x position).
    t_axis = torch.arange(0, int(tau), device=dev, dtype=dtype) * float(dt) * float(lambda_max)
    x_axis = torch.linspace(0.0, float(L), int(Q), device=dev, dtype=dtype)
    extent = [float(t_axis[0]), float(t_axis[-1]), float(x_axis[0]), float(x_axis[-1])]

    fig, axs = plt.subplots(4, 1, figsize=(7.2, 8.5), constrained_layout=True)
    for ax, data, title in zip(
        axs,
        [true, pred, err_res, err_int],
        ["(a) True KS", "(b) Reservoir prediction", "(c) True - pred", "(d) True - KS(init=pred[0])"],
    ):
        im = ax.imshow(
            data.detach().cpu().numpy(),
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        ax.set_title(title)
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axs[-1].set_xlabel(r"$\Lambda_{max} t$")

    return fig, {
        "true": true,
        "pred": pred,
        "err_reservoir": err_res,
        "err_integrated": err_int,
        "t": t_axis,
        "x": x_axis,
    }


@torch.no_grad()
def reproduce_prl_fig5a(
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    seed_data: int = 0,
    # KS settings
    mu: float = 0.01,
    wavelength: float = 100.0,
    dt: float = 0.25,
    # Evaluation settings
    tau: int = 1000,
    K: int = 30,
    epsilon: int = 10,
    num_trials: int = 10,
    # Reservoir settings
    locality: int = 6,
    approx_reservoir_size: int = 5000,
    radius: float = 0.6,
    degree: float = 3.0,
    beta: float = 1e-4,
    sigma: float = 1.0,
    discard_length: int = 1000,
    train_length: int = 70_000,
    # Fig.5(a) sizes
    L_values: Sequence[int] = (200, 400, 800, 1600),
    lambda_max: Optional[Dict[int, float]] = None,
):
    """Reproduce the PRL Fig.5(a) benchmarking curves.

    The paper keeps L/g fixed at 200/64, i.e. g scales with L.
    We also keep the spatial resolution fixed by scaling Q proportionally
    with L (Q=512 at L=200).

    Returns:
        fig, results dict keyed by L, each containing the benchmark dict.
    """
    import matplotlib.pyplot as plt

    dev = _as_device(device)

    if lambda_max is None:
        lambda_max = {200: 0.09, 400: 0.09, 800: 0.10, 1600: 0.10}

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax.set_xlabel(r"$\Lambda_{max} t$")
    ax.set_ylabel("RMSE")

    results: Dict[int, Dict[str, torch.Tensor]] = {}

    for L in L_values:
        g = int(round(64 * (int(L) / 200)))
        Q = int(round(512 * (int(L) / 200)))
        if Q % 2 != 0:
            Q += 1

        need = int(discard_length + train_length)
        total_steps = int(need + K * tau)
        u = generate_ks_dataset(
            Q=Q,
            L=float(L),
            mu=float(mu),
            wavelength=float(wavelength),
            dt=float(dt),
            n_steps=total_steps,
            burn_in=0,
            seed=int(seed_data),
            device=dev,
            dtype=dtype,
        )

        params = KSParallelParams(
            Q=Q,
            g=g,
            locality=locality,
            approx_reservoir_size=approx_reservoir_size,
            radius=radius,
            degree=degree,
            beta=beta,
            sigma=sigma,
            discard_length=discard_length,
            train_length=train_length,
            predict_length=tau,
            jobid=1,
        )

        bench = benchmark_parallel_reservoir(
            u,
            params=params,
            dt=dt,
            lambda_max=float(lambda_max[int(L)]),
            tau=tau,
            K=K,
            epsilon=epsilon,
            num_trials=num_trials,
            identical_reservoirs=False,
            device=dev,
            dtype=dtype,
        )
        results[int(L)] = bench
        ax.plot(bench["t"].detach().cpu().numpy(), bench["rmse_mean"].detach().cpu().numpy(), label=f"L = {L}")

    ax.legend(frameon=False)
    ax.set_title("PRL Fig.5(a) style: RMSE vs time for different L")
    return fig, results


@torch.no_grad()
def reproduce_prl_fig5b(
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    seed_data: int = 0,
    # KS settings
    L: float = 200.0,
    Q: int = 512,
    mu: float = 0.01,
    wavelength: float = 100.0,
    dt: float = 0.25,
    lambda_max: float = 0.09,
    # Evaluation settings
    tau: int = 1000,
    K: int = 30,
    epsilon: int = 10,
    num_trials: int = 10,
    # Reservoir settings
    locality: int = 6,
    approx_reservoir_size: int = 5000,
    radius: float = 0.6,
    degree: float = 3.0,
    beta: float = 1e-4,
    sigma: float = 1.0,
    discard_length: int = 1000,
    train_length: int = 70_000,
    g_values: Sequence[int] = (8, 16, 32, 64),
):
    """Reproduce the PRL Fig.5(b) benchmarking curves (vary g)."""
    import matplotlib.pyplot as plt

    dev = _as_device(device)
    need = int(discard_length + train_length)
    total_steps = int(need + K * tau)
    u = generate_ks_dataset(
        Q=int(Q),
        L=float(L),
        mu=float(mu),
        wavelength=float(wavelength),
        dt=float(dt),
        n_steps=total_steps,
        burn_in=0,
        seed=int(seed_data),
        device=dev,
        dtype=dtype,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax.set_xlabel(r"$\Lambda_{max} t$")
    ax.set_ylabel("RMSE")

    results: Dict[int, Dict[str, torch.Tensor]] = {}
    for g in g_values:
        params = KSParallelParams(
            Q=int(Q),
            g=int(g),
            locality=int(locality),
            approx_reservoir_size=int(approx_reservoir_size),
            radius=float(radius),
            degree=float(degree),
            beta=float(beta),
            sigma=float(sigma),
            discard_length=int(discard_length),
            train_length=int(train_length),
            predict_length=int(tau),
            jobid=1,
        )

        bench = benchmark_parallel_reservoir(
            u,
            params=params,
            dt=float(dt),
            lambda_max=float(lambda_max),
            tau=int(tau),
            K=int(K),
            epsilon=int(epsilon),
            num_trials=int(num_trials),
            identical_reservoirs=False,
            device=dev,
            dtype=dtype,
        )
        results[int(g)] = bench
        ax.plot(bench["t"].detach().cpu().numpy(), bench["rmse_mean"].detach().cpu().numpy(), label=f"g = {g}")

    ax.legend(frameon=False)
    ax.set_title("PRL Fig.5(b) style: RMSE vs time for different g")
    return fig, results


@torch.no_grad()
def reproduce_prl_fig6(
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    seed_data: int = 0,
    # KS settings
    L: float = 200.0,
    Q: int = 512,
    wavelength: float = 100.0,
    dt: float = 0.25,
    lambda_max: float = 0.09,
    # Evaluation settings
    tau: int = 1000,
    K: int = 30,
    epsilon: int = 10,
    num_trials: int = 10,
    # Reservoir settings
    g: int = 64,
    locality: int = 6,
    approx_reservoir_size: int = 5000,
    radius: float = 0.6,
    degree: float = 3.0,
    beta: float = 1e-4,
    sigma: float = 1.0,
    discard_length: int = 1000,
    train_length: int = 70_000,
):
    """Reproduce PRL Fig.6: identical vs independently trained reservoirs.

    Returns:
        fig_mu0, fig_mu001, results dict.
    """
    import matplotlib.pyplot as plt

    dev = _as_device(device)
    need = int(discard_length + train_length)
    total_steps = int(need + K * tau)

    results: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    def run(mu: float) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        u = generate_ks_dataset(
            Q=int(Q),
            L=float(L),
            mu=float(mu),
            wavelength=float(wavelength),
            dt=float(dt),
            n_steps=total_steps,
            burn_in=0,
            seed=int(seed_data),
            device=dev,
            dtype=dtype,
        )
        params = KSParallelParams(
            Q=int(Q),
            g=int(g),
            locality=int(locality),
            approx_reservoir_size=int(approx_reservoir_size),
            radius=float(radius),
            degree=float(degree),
            beta=float(beta),
            sigma=float(sigma),
            discard_length=int(discard_length),
            train_length=int(train_length),
            predict_length=int(tau),
            jobid=1,
        )
        bench_identical = benchmark_parallel_reservoir(
            u,
            params=params,
            dt=float(dt),
            lambda_max=float(lambda_max),
            tau=int(tau),
            K=int(K),
            epsilon=int(epsilon),
            num_trials=int(num_trials),
            identical_reservoirs=True,
            device=dev,
            dtype=dtype,
        )
        bench_indep = benchmark_parallel_reservoir(
            u,
            params=params,
            dt=float(dt),
            lambda_max=float(lambda_max),
            tau=int(tau),
            K=int(K),
            epsilon=int(epsilon),
            num_trials=int(num_trials),
            identical_reservoirs=False,
            device=dev,
            dtype=dtype,
        )
        return bench_identical, bench_indep

    # (a) mu = 0
    b_id_0, b_in_0 = run(0.0)
    results["mu=0"] = {"identical": b_id_0, "independent": b_in_0}
    fig0, ax0 = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax0.plot(b_id_0["t"].detach().cpu().numpy(), b_id_0["rmse_mean"].detach().cpu().numpy(), label="all identical")
    ax0.plot(b_in_0["t"].detach().cpu().numpy(), b_in_0["rmse_mean"].detach().cpu().numpy(), label="trained independently")
    ax0.set_title("PRL Fig.6(a) style (mu = 0)")
    ax0.set_xlabel(r"$\Lambda_{max} t$")
    ax0.set_ylabel("RMSE")
    ax0.legend(frameon=False)

    # (b) mu = 0.01
    b_id_1, b_in_1 = run(0.01)
    results["mu=0.01"] = {"identical": b_id_1, "independent": b_in_1}
    fig1, ax1 = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax1.plot(b_id_1["t"].detach().cpu().numpy(), b_id_1["rmse_mean"].detach().cpu().numpy(), label="all identical")
    ax1.plot(b_in_1["t"].detach().cpu().numpy(), b_in_1["rmse_mean"].detach().cpu().numpy(), label="trained independently")
    ax1.set_title("PRL Fig.6(b) style (mu = 0.01)")
    ax1.set_xlabel(r"$\Lambda_{max} t$")
    ax1.set_ylabel("RMSE")
    ax1.legend(frameon=False)

    return fig0, fig1, results
