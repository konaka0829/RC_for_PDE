from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from .ks_reservoir import KSBasicSingleReservoir
from .benchmark import benchmark_parallel_reservoir
from .ks_parallel_reservoir import KSParallelParams, KSParallelReservoir
from .solver import _as_device, generate_ks_dataset, ks_solve_etdrk4_forced, kursiv_solve_etdrk4


def reproduce_prl_figure2(
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    seed_data: Optional[int] = None,
    seed_A: Optional[int] = None,
    # KS parameters (PRL Fig.2: L=22, Q=64, mu=0)
    ks_N: int = 64,
    ks_d: float = 22.0,
    ks_dt: float = 0.25,
    ks_steps: int = 100_000,
    # Reservoir parameters
    approx_reservoir_size: int = 5000,
    radius: float = 0.6,
    degree: float = 3.0,
    sigma: float = 1.0,
    beta: float = 1e-4,
    train_length: int = 70_000,
    predict_length: int = 1_000,
    lambda_max: float = 0.05,
    # Plot settings
    xlim: Tuple[float, float] = (0.0, 12.0),
):
    """Reproduce Fig. 2 style plots (actual / prediction / error) for the KS demo.

    Returns:
        fig, (actual, pred, err), (t_axis, x_axis)
    """
    import matplotlib.pyplot as plt

    dev = _as_device(device)

    if seed_data is not None:
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed_data))
        init = 0.6 * (-1.0 + 2.0 * torch.rand(ks_N, generator=gen, device=dev, dtype=dtype))
    else:
        init = 0.6 * (-1.0 + 2.0 * torch.rand(ks_N, device=dev, dtype=dtype))

    uu = kursiv_solve_etdrk4(init, dt=ks_dt, n_steps=ks_steps, d=ks_d, device=dev, dtype=dtype)
    data = uu.T  # [N, T]

    if train_length + predict_length > data.shape[1]:
        raise ValueError("Need ks_steps >= train_length + predict_length")

    num_inputs = ks_N
    reservoir_size = (approx_reservoir_size // num_inputs) * num_inputs
    model = KSBasicSingleReservoir(
        num_inputs=num_inputs,
        reservoir_size=reservoir_size,
        radius=radius,
        degree=degree,
        sigma=sigma,
        beta=beta,
        seed_A=seed_A,
        device=dev,
        dtype=dtype,
    )
    model.fit(data[:, :train_length], chunk_size=2048)
    pred, _ = model.predict(predict_length)

    actual = data[:, train_length : train_length + predict_length]
    err = pred - actual

    t_axis = torch.arange(1, predict_length + 1, device=dev, dtype=dtype) * float(ks_dt) * float(lambda_max)
    x_axis = torch.linspace(0.0, float(ks_d), steps=ks_N + 1, device=dev, dtype=dtype)[:-1]

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    extent = [float(t_axis[0]), float(t_axis[-1]), float(x_axis[0]), float(x_axis[-1])]
    im0 = axs[0].imshow(actual.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[0].set_title("Actual")
    axs[0].set_xlabel(r"$\Lambda_{max} t$")
    axs[0].set_ylabel("x")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[1].set_title("Prediction")
    axs[1].set_xlabel(r"$\Lambda_{max} t$")
    axs[1].set_ylabel("x")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(err.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[2].set_title("Error (prediction - actual)")
    axs[2].set_xlabel(r"$\Lambda_{max} t$")
    axs[2].set_ylabel("x")
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlim(*xlim)

    plt.set_cmap("jet")

    return fig, (actual, pred, err), (t_axis, x_axis)


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
    integ_scaled = float(sigma) * integ.T
    err_int = true - integ_scaled

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
    """Reproduce the PRL Fig.5(a) benchmarking curves."""
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
    """Reproduce PRL Fig.6: identical vs independently trained reservoirs."""
    import matplotlib.pyplot as plt

    dev = _as_device(device)
    need = int(discard_length + train_length)
    total_steps = int(need + K * tau)

    results: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    def run(mu: float):
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

    b_id_0, b_in_0 = run(0.0)
    results["mu=0"] = {"identical": b_id_0, "independent": b_in_0}
    fig0, ax0 = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
    ax0.plot(b_id_0["t"].detach().cpu().numpy(), b_id_0["rmse_mean"].detach().cpu().numpy(), label="all identical")
    ax0.plot(b_in_0["t"].detach().cpu().numpy(), b_in_0["rmse_mean"].detach().cpu().numpy(), label="trained independently")
    ax0.set_title("PRL Fig.6(a) style (mu = 0)")
    ax0.set_xlabel(r"$\Lambda_{max} t$")
    ax0.set_ylabel("RMSE")
    ax0.legend(frameon=False)

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
