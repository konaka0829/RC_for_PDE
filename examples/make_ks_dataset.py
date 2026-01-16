import argparse
from pathlib import Path

import numpy as np

from torchesn.utils.datasets_ks import generate_or_load_ks_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate KS dataset cache.")
    parser.add_argument("--output", type=Path, default=Path("examples/datasets/ks_dataset.npz"))
    parser.add_argument("--L", type=float, default=22.0)
    parser.add_argument("--Q", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--total-steps", type=int, default=600)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = generate_or_load_ks_dataset(
        args.output,
        L=args.L,
        Q=args.Q,
        dt=args.dt,
        mu=args.mu,
        lam=args.lam,
        total_steps=args.total_steps,
        burn_in=args.burn_in,
        seed=args.seed,
        dtype=np.float32,
    )
    print(f"Saved KS dataset to {args.output} with shape {data.shape}.")


if __name__ == "__main__":
    main()
