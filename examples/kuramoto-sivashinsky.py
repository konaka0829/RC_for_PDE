"""
Kuramoto-Sivashinsky (KS) data generation and time-series prediction using a
MATLAB-compatible single-reservoir ESN.

This script:
1. Generates KS data with the ETDRK4 solver.
2. Trains a KSBasicSingleReservoir on the training segment.
3. Runs closed-loop prediction for the forecast horizon.
4. Saves actual/pred/error plots to disk.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import torch
import matplotlib.pyplot as plt

from torchesn.ks import KSBasicSingleReservoir, kursiv_solve_etdrk4


def save_figure(fig, output_path_base):
    output_dir = os.path.dirname(output_path_base)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{output_path_base}.{ext}", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="KS data generation and time-series prediction with ESN"
    )
    parser.add_argument("--ks-n", type=int, default=64)
    parser.add_argument("--ks-d", type=float, default=22.0)
    parser.add_argument("--ks-dt", type=float, default=0.25)
    parser.add_argument("--ks-steps", type=int, default=1000)
    parser.add_argument("--train-length", type=int, default=700)
    parser.add_argument("--predict-length", type=int, default=100)
    parser.add_argument("--reservoir-size", type=int, default=512)
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--degree", type=float, default=3.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--seed-data", type=int, default=1)
    parser.add_argument("--seed-A", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float", "double"], default="double")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = torch.float64 if args.dtype == "double" else torch.float32

    if args.train_length + args.predict_length > args.ks_steps:
        raise ValueError("ks-steps must be >= train-length + predict-length")
    if args.reservoir_size % args.ks_n != 0:
        raise ValueError("reservoir-size must be divisible by ks-n")

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed_data))
    init = 0.6 * (-1.0 + 2.0 * torch.rand(args.ks_n, generator=gen, device=device, dtype=dtype))

    uu = kursiv_solve_etdrk4(
        init,
        dt=args.ks_dt,
        n_steps=args.ks_steps,
        d=args.ks_d,
        device=device,
        dtype=dtype,
    )
    data = uu.T  # [N, T]

    model = KSBasicSingleReservoir(
        num_inputs=args.ks_n,
        reservoir_size=args.reservoir_size,
        radius=args.radius,
        degree=args.degree,
        sigma=args.sigma,
        beta=args.beta,
        seed_A=args.seed_A,
        device=device,
        dtype=dtype,
    )
    model.fit(data[:, : args.train_length], chunk_size=2048)
    pred, _ = model.predict(args.predict_length)

    actual = data[:, args.train_length : args.train_length + args.predict_length]
    err = pred - actual

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    extent = [
        0.0,
        float(args.predict_length * args.ks_dt),
        0.0,
        float(args.ks_d),
    ]

    im0 = axs[0].imshow(actual.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[0].set_title("Actual")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[1].set_title("Prediction")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("x")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(err.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[2].set_title("Error (prediction - actual)")
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("x")
    fig.colorbar(im2, ax=axs[2])

    plt.set_cmap("jet")
    output_base = os.path.join(os.path.dirname(__file__), "figures", "ks_prediction")
    save_figure(fig, output_base)


if __name__ == "__main__":
    main()
