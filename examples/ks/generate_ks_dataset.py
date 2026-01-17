"""Generate KS datasets for parallel reservoir computing experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchesn.pde.ks_etdrk4 import simulate_ks_etdrk4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KS equation dataset and save as NPZ.")
    parser.add_argument("--out", required=True, help="Output .npz file path")
    parser.add_argument("--L", type=float, default=50.0, help="Domain length")
    parser.add_argument("--Q", type=int, default=32, help="Number of spatial grid points")
    parser.add_argument("--dt", type=float, default=0.25, help="Time step")
    parser.add_argument("--mu", type=float, default=0.01, help="Forcing amplitude")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=25.0, help="Forcing wavelength")
    parser.add_argument("--total-steps", type=int, default=2000, help="Total time steps")
    parser.add_argument("--train-steps", type=int, default=1500, help="Training steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Output dtype",
    )
    parser.add_argument("--u0-scale", type=float, default=0.6, help="Initial condition scale")
    parser.add_argument(
        "--substeps",
        type=int,
        default=4,
        help="ETDRK4 substeps per output step (improves stability for coarse grids)",
    )
    parser.add_argument(
        "--filter-strength",
        type=float,
        default=100.0,
        help="Exponential spectral filter strength (0 disables filtering)",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=8,
        help="Exponential spectral filter order (higher is sharper)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.train_steps >= args.total_steps:
        raise ValueError("train_steps must be less than total_steps")

    dtype = np.dtype(args.dtype)

    series = simulate_ks_etdrk4(
        L=args.L,
        Q=args.Q,
        dt=args.dt,
        n_steps=args.total_steps,
        mu=args.mu,
        lambda_=args.lambda_,
        seed=args.seed,
        u0_scale=args.u0_scale,
        dtype=dtype,
        n_substeps=args.substeps,
        filter_strength=args.filter_strength,
        filter_order=args.filter_order,
    )
    if not np.isfinite(series).all():
        bad_idx = np.argwhere(~np.isfinite(series))[0]
        raise RuntimeError(f"Non-finite values detected at t={bad_idx[0]}, x={bad_idx[1]}")

    train_u = series[: args.train_steps]
    test_u = series[args.train_steps :]

    meta = {
        "L": args.L,
        "Q": args.Q,
        "dt": args.dt,
        "mu": args.mu,
        "lambda": args.lambda_,
        "total_steps": args.total_steps,
        "train_steps": args.train_steps,
        "test_steps": args.total_steps - args.train_steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "u0_scale": args.u0_scale,
        "substeps": args.substeps,
        "filter_strength": args.filter_strength,
        "filter_order": args.filter_order,
    }

    meta_json = json.dumps(meta, sort_keys=True)
    np.savez(out_path, train_u=train_u, test_u=test_u, meta_json=meta_json)


if __name__ == "__main__":
    main()
