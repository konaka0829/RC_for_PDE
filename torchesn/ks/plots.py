from __future__ import annotations

from typing import Optional, Tuple

import torch

from .ks_reservoir import KSBasicSingleReservoir
from .solver import _as_device, kursiv_solve_etdrk4


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
