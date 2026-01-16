import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchesn.nn import ParallelESN
from torchesn.utils.datasets_ks import generate_or_load_ks_dataset
from torchesn.utils.kuramoto_sivashinsky import simulate_ks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_figure(fig, out_dir, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")


def apply_quick_defaults(args):
    args.L = 22.0
    args.Q = 64
    args.dt = 0.25
    args.mu = 0.0
    args.lam = 100.0
    args.burn_in = 200
    args.g = 8
    args.l = 2
    args.hidden_size = 200
    args.spectral_radius = 0.6
    args.lambda_reg = max(args.lambda_reg, 1e-2)
    args.T_train = 400
    args.K = 2
    args.tau = 50
    args.epsilon = 10
    args.n_trials = 1
    args.T_eval = max(args.T_eval, args.K * args.tau + args.epsilon + 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ParallelESN on KS data.")

    parser.add_argument("--L", type=float, default=200.0)
    parser.add_argument("--Q", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=0)

    parser.add_argument("--T-train", type=int, default=70000)
    parser.add_argument("--T-eval", type=int, default=0)
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--tau", type=int, default=1000)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=10)

    parser.add_argument("--g", type=int, default=64)
    parser.add_argument("--l", type=int, default=6)
    parser.add_argument("--hidden-size", "--Dr", type=int, default=5000)
    parser.add_argument("--spectral-radius", "--rho", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--kappa", type=int, default=3)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--leaking-rate", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--readout-training", type=str, default="cholesky")
    parser.add_argument("--readout-features", type=str, default="linear_and_square")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float")

    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="examples/figures/ks_eval")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype):
    if dtype == "double":
        return torch.float64
    return torch.float32


def compute_lengths(args):
    if args.T_eval <= 0:
        args.T_eval = args.K * args.tau + args.epsilon + 1
    min_eval = args.K * args.tau + args.epsilon + 1
    if args.T_eval < min_eval:
        args.T_eval = min_eval
    if args.total_steps <= 0:
        args.total_steps = args.burn_in + args.T_train + args.T_eval


def load_or_simulate(args):
    if args.cache_path:
        return generate_or_load_ks_dataset(
            args.cache_path,
            L=args.L,
            Q=args.Q,
            dt=args.dt,
            mu=args.mu,
            lam=args.lam,
            total_steps=args.total_steps - args.burn_in,
            burn_in=args.burn_in,
            seed=args.seed,
            dtype=np.float32,
        )
    return simulate_ks(
        L=args.L,
        Q=args.Q,
        dt=args.dt,
        n_steps=args.total_steps - args.burn_in,
        mu=args.mu,
        lam=args.lam,
        seed=args.seed,
        burn_in=args.burn_in,
        dtype=np.float32,
    )


def build_model(args, device, seed):
    return ParallelESN(
        Q=args.Q,
        g=args.g,
        l=args.l,
        hidden_size=args.hidden_size,
        spectral_radius=args.spectral_radius,
        leaking_rate=args.leaking_rate,
        density=args.density,
        lambda_reg=args.lambda_reg,
        readout_training=args.readout_training,
        readout_features=args.readout_features,
        w_io=False,
        translation_invariant=False,
        mu=args.mu,
        seed=seed,
    ).to(device)


def evaluate_trial(args, model, u_train, u_eval):
    model.fit(u_train, washout=0)
    rmse_sum = torch.zeros(args.tau, device=u_eval.device)
    for k in range(args.K):
        start = k * args.tau
        hist = u_eval[start : start + args.epsilon + 1]
        pred = model.predict(hist, steps=args.tau, epsilon=args.epsilon)
        true = u_eval[start + args.epsilon + 1 : start + args.epsilon + 1 + args.tau]
        rmse = torch.sqrt(torch.mean((true - pred) ** 2, dim=1))
        rmse_sum += rmse
    return rmse_sum / args.K


def main():
    args = parse_args()
    if args.quick:
        apply_quick_defaults(args)
    if args.Q % args.g != 0:
        raise ValueError("Q must be divisible by g.")
    compute_lengths(args)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    data = load_or_simulate(args)

    u_train = torch.tensor(data[: args.T_train], dtype=dtype, device=device)
    u_eval = torch.tensor(data[args.T_train : args.T_train + args.T_eval], dtype=dtype, device=device)

    rmse_trials = []
    for trial in range(args.n_trials):
        seed = args.seed + trial
        model = build_model(args, device, seed)
        rmse_trials.append(evaluate_trial(args, model, u_train, u_eval))

    rmse_curve = torch.stack(rmse_trials, dim=0).mean(dim=0)
    rmse_np = rmse_curve.detach().cpu().numpy().tolist()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(rmse_curve.detach().cpu().numpy())
    plt.xlabel("t")
    plt.ylabel("RMSE")
    plt.tight_layout()
    save_figure(fig, out_dir, "rmse_curve")
    plt.close(fig)

    result = {
        "params": vars(args),
        "rmse_curve": rmse_np,
        "rmse_mean": float(np.mean(rmse_np)),
        "rmse_start": float(rmse_np[0]),
        "rmse_end": float(rmse_np[-1]),
    }
    with open(out_dir / "rmse_curve.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        "RMSE summary: "
        f"start={result['rmse_start']:.6f}, "
        f"end={result['rmse_end']:.6f}, "
        f"mean={result['rmse_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
