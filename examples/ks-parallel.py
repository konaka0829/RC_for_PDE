"""
Minimal CLI example for running the KS parallel reservoir pipeline.

Steps:
1. Generate KS dataset.
2. Fit parallel reservoir.
3. Run closed-loop prediction.
4. Evaluate RMSE curve with benchmark helper.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import torch

from torchesn.ks import (
    KSParallelParams,
    KSParallelReservoir,
    benchmark_parallel_reservoir,
    generate_ks_dataset,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KS parallel reservoir demo")
    parser.add_argument("--Q", type=int, default=64)
    parser.add_argument("--L", type=float, default=200.0)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--wavelength", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float", "double"], default="double")
    parser.add_argument("--g", type=int, default=8)
    parser.add_argument("--locality", type=int, default=2)
    parser.add_argument("--approx-reservoir-size", type=int, default=1000)
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--degree", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--discard-length", type=int, default=100)
    parser.add_argument("--train-length", type=int, default=1000)
    parser.add_argument("--predict-length", type=int, default=200)
    parser.add_argument("--jobid", type=int, default=1)
    parser.add_argument("--tau", type=int, default=200)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    dtype = torch.float64 if args.dtype == "double" else torch.float32

    u = generate_ks_dataset(
        Q=args.Q,
        L=args.L,
        mu=args.mu,
        wavelength=args.wavelength,
        dt=args.dt,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        seed=args.seed,
        device=device,
    ).to(dtype)

    params = KSParallelParams(
        Q=args.Q,
        g=args.g,
        locality=args.locality,
        approx_reservoir_size=args.approx_reservoir_size,
        radius=args.radius,
        degree=args.degree,
        beta=args.beta,
        sigma=args.sigma,
        discard_length=args.discard_length,
        train_length=args.train_length,
        predict_length=args.predict_length,
        jobid=args.jobid,
    )

    model = KSParallelReservoir(params, device=device, dtype=dtype)
    model.fit(u[:, : params.discard_length + params.train_length])

    pred = model.predict_one_interval(
        u,
        warmup_start=params.discard_length + params.train_length - 10,
        sync_length=10,
        predict_length=params.predict_length,
    )

    bench = benchmark_parallel_reservoir(
        u,
        params=params,
        tau=args.tau,
        K=args.K,
        epsilon=args.epsilon,
        num_trials=args.num_trials,
    )

    print("Prediction shape:", pred.shape)
    print("RMSE mean shape:", bench["rmse_mean"].shape)


if __name__ == "__main__":
    main()
