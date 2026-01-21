"""CLI for running KS reservoir predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch

from rc_for_pde.ks.solver import generate_ks_dataset, kursiv_solve_etdrk4
from rc_for_pde.reservoirs.parallel import KSParallelParams, KSParallelReservoir
from rc_for_pde.reservoirs.single import KSBasicSingleReservoir

TimeAxis = Literal["scaled", "discrete"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS reservoir prediction runner")
    parser.add_argument("--mode", choices=["single", "parallel"], default="single")
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for data generation")
    parser.add_argument("--plot", action="store_true", help="save prediction plot")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("prediction.png"),
        help="output path for plot (when --plot)",
    )
    parser.add_argument(
        "--time-axis",
        choices=["scaled", "discrete"],
        default="scaled",
        help="plot time axis: Lyapunov-scaled or discrete steps",
    )
    parser.add_argument("--lambda-max", type=float, default=0.05, help="largest Lyapunov exponent")

    # Shared reservoir hyperparameters.
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--degree", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=1.0)

    # Single-reservoir parameters.
    parser.add_argument("--ks-n", type=int, default=64, help="spatial grid size for KS (single)")
    parser.add_argument("--ks-d", type=float, default=22.0, help="domain length for KS (single)")
    parser.add_argument("--ks-dt", type=float, default=0.25, help="time step for KS (single)")
    parser.add_argument("--ks-steps", type=int, default=100_000, help="total steps for KS (single)")
    parser.add_argument("--reservoir-size", type=int, default=5000, help="approx reservoir size (single)")
    parser.add_argument("--train-length", type=int, default=70_000, help="training length")
    parser.add_argument("--predict-length", type=int, default=1_000, help="prediction length")

    # Parallel-reservoir parameters.
    parser.add_argument("--Q", type=int, default=512, help="spatial grid size (parallel)")
    parser.add_argument("--L", type=float, default=200.0, help="domain length (parallel)")
    parser.add_argument("--mu", type=float, default=0.01, help="forcing strength")
    parser.add_argument("--wavelength", type=float, default=100.0, help="forcing wavelength")
    parser.add_argument("--dt", type=float, default=0.25, help="time step (parallel)")
    parser.add_argument("--burn-in", type=int, default=0, help="burn-in steps for parallel data")
    parser.add_argument("--discard-length", type=int, default=1000, help="discard length (parallel)")
    parser.add_argument("--parallel-train-length", type=int, default=79_000, help="training length (parallel)")
    parser.add_argument("--parallel-predict-length", type=int, default=2999, help="prediction length (parallel)")
    parser.add_argument("--g", type=int, default=64, help="number of reservoirs (parallel)")
    parser.add_argument("--locality", type=int, default=6, help="locality overlap (parallel)")
    parser.add_argument("--sync-length", type=int, default=32, help="warm-up length (parallel)")
    parser.add_argument(
        "--approx-reservoir-size",
        type=int,
        default=5000,
        help="approx reservoir size per input window (parallel)",
    )

    return parser.parse_args()


def _time_axis(
    length: int,
    dt: float,
    lambda_max: float,
    mode: TimeAxis,
) -> Tuple[np.ndarray, str]:
    steps = np.arange(1, length + 1, dtype=float)
    if mode == "scaled":
        return steps * dt * lambda_max, r"$\Lambda_{max} t$"
    return steps, "t (step)"


def _plot_prediction(
    actual: torch.Tensor,
    pred: torch.Tensor,
    err: torch.Tensor,
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    xlabel: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    extent = [float(t_axis[0]), float(t_axis[-1]), float(x_axis[0]), float(x_axis[-1])]

    im0 = axs[0].imshow(actual, aspect="auto", origin="lower", extent=extent)
    axs[0].set_title("Actual")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel("x")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred, aspect="auto", origin="lower", extent=extent)
    axs[1].set_title("Prediction")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("x")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(err, aspect="auto", origin="lower", extent=extent)
    axs[2].set_title("Error (prediction - actual)")
    axs[2].set_xlabel(xlabel)
    axs[2].set_ylabel("x")
    fig.colorbar(im2, ax=axs[2])

    plt.set_cmap("jet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)


@torch.no_grad()
def _run_single(args: argparse.Namespace) -> None:
    dev = torch.device(args.device)
    gen = torch.Generator(device=dev)
    if args.seed is not None:
        gen.manual_seed(int(args.seed))

    init = 0.6 * (-1.0 + 2.0 * torch.rand(args.ks_n, generator=gen, device=dev, dtype=torch.float64))
    uu = kursiv_solve_etdrk4(init, dt=args.ks_dt, n_steps=args.ks_steps, d=args.ks_d, device=dev)
    data = uu.T

    if args.train_length + args.predict_length > data.shape[1]:
        raise ValueError("Need ks_steps >= train_length + predict_length")

    reservoir_size = (args.reservoir_size // args.ks_n) * args.ks_n
    model = KSBasicSingleReservoir(
        num_inputs=args.ks_n,
        reservoir_size=reservoir_size,
        radius=args.radius,
        degree=args.degree,
        sigma=args.sigma,
        beta=args.beta,
        device=dev,
    )
    model.fit(data[:, : args.train_length], chunk_size=2048)
    pred, _ = model.predict(args.predict_length)

    actual = data[:, args.train_length : args.train_length + args.predict_length]
    err = pred - actual

    t_axis, xlabel = _time_axis(args.predict_length, args.ks_dt, args.lambda_max, args.time_axis)
    x_axis = np.linspace(0.0, float(args.ks_d), num=args.ks_n + 1)[:-1]

    if args.plot:
        _plot_prediction(
            actual.cpu().numpy(),
            pred.cpu().numpy(),
            err.cpu().numpy(),
            t_axis,
            x_axis,
            xlabel,
            args.plot_path,
        )


@torch.no_grad()
def _run_parallel(args: argparse.Namespace) -> None:
    dev = torch.device(args.device)
    total_steps = args.discard_length + args.parallel_train_length + args.sync_length + args.parallel_predict_length

    data = generate_ks_dataset(
        Q=args.Q,
        L=args.L,
        mu=args.mu,
        wavelength=args.wavelength,
        dt=args.dt,
        n_steps=total_steps,
        burn_in=args.burn_in,
        seed=args.seed,
        device=dev,
    )

    params = KSParallelParams(
        Q=args.Q,
        g=args.g,
        locality=args.locality,
        approx_reservoir_size=args.approx_reservoir_size,
        radius=args.radius,
        degree=args.degree,
        beta=args.beta,
        discard_length=args.discard_length,
        train_length=args.parallel_train_length,
        predict_length=args.parallel_predict_length,
        sigma=args.sigma,
    )

    model = KSParallelReservoir(params, device=dev)
    model.fit(data[:, : args.discard_length + args.parallel_train_length])

    warmup_start = args.discard_length + args.parallel_train_length - args.sync_length
    pred = model.predict_one_interval(
        data,
        warmup_start=warmup_start,
        sync_length=args.sync_length,
        predict_length=args.parallel_predict_length,
    )

    actual = data[:, warmup_start + args.sync_length : warmup_start + args.sync_length + args.parallel_predict_length]
    err = pred - actual

    t_axis, xlabel = _time_axis(args.parallel_predict_length, args.dt, args.lambda_max, args.time_axis)
    x_axis = np.linspace(0.0, float(args.L), num=args.Q + 1)[:-1]

    if args.plot:
        _plot_prediction(
            actual.cpu().numpy(),
            pred.cpu().numpy(),
            err.cpu().numpy(),
            t_axis,
            x_axis,
            xlabel,
            args.plot_path,
        )


def main() -> None:
    args = _parse_args()
    if args.mode == "single":
        _run_single(args)
    else:
        _run_parallel(args)


if __name__ == "__main__":
    main()
