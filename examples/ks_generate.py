#!/usr/bin/env python
"""CLI for generating Kuramoto–Sivashinsky time series."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchesn.pde import simulate_ks_etdrk4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Kuramoto–Sivashinsky time series data."
    )
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--L", type=float, default=22.0)
    parser.add_argument("--transient", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--lambda", dest="forcing_wavelength", type=float)
    parser.add_argument("--dealias", action="store_true")
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--divergence-threshold", type=float, default=1e6)
    parser.add_argument(
        "--dtype",
        choices=["float", "double"],
        default="double",
        help="Floating point precision",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a short run for quick checks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.quick:
        args.n_steps = 300
        args.transient = 50
        args.save_every = 1

    dtype = np.float64 if args.dtype == "double" else np.float32

    result = simulate_ks_etdrk4(
        n_steps=args.n_steps,
        dt=args.dt,
        n_grid=args.n_grid,
        domain_length=args.L,
        transient=args.transient,
        save_every=args.save_every,
        seed=args.seed,
        mu=args.mu,
        forcing_wavelength=args.forcing_wavelength,
        dealias=args.dealias,
        check_interval=args.check_interval,
        divergence_threshold=args.divergence_threshold,
        dtype=dtype,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(result["meta"], ensure_ascii=False, sort_keys=True)
    np.savez(output_path, u=result["u"], x=result["x"], t=result["t"], meta=meta_json)

    u = result["u"]
    stats = (
        float(np.min(u)),
        float(np.max(u)),
        float(np.mean(u)),
        float(np.std(u)),
    )
    finite = bool(np.isfinite(u).all())

    print(f"Saved: {output_path}")
    print(f"u shape: {u.shape}")
    print(
        "u stats: min={:.6g} max={:.6g} mean={:.6g} std={:.6g} finite={}"
        .format(*stats, finite)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
