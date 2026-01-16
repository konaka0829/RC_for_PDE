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
from torchesn.utils.experiment_profiles import (
    config_to_namespace,
    resolve_experiment_config,
)
from torchesn.utils.kuramoto_sivashinsky import simulate_ks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_figure(fig, out_dir, stem):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ParallelESN on KS data.")

    parser.add_argument("--L", type=float, default=None)
    parser.add_argument("--Q", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)

    parser.add_argument("--T-train", type=int, default=None)
    parser.add_argument("--T-eval", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--epsilon", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)

    parser.add_argument("--g", type=int, default=None)
    parser.add_argument("--l", type=int, default=None)
    parser.add_argument("--hidden-size", "--Dr", type=int, default=None)
    parser.add_argument("--spectral-radius", "--rho", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--kappa", type=int, default=None)
    parser.add_argument("--density", type=float, default=None)
    parser.add_argument("--leaking-rate", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=None)
    parser.add_argument("--readout-training", type=str, default=None)
    parser.add_argument("--readout-features", type=str, default=None)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float")

    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="examples/figures/ks_eval")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--paper-defaults", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype):
    if dtype == "double":
        return torch.float64
    return torch.float32


def load_or_simulate(args, config):
    if args.cache_path:
        return generate_or_load_ks_dataset(
            args.cache_path,
            L=config.L,
            Q=config.Q,
            dt=config.dt,
            mu=config.mu,
            lam=config.lam,
            total_steps=config.total_steps - config.burn_in,
            burn_in=config.burn_in,
            seed=args.seed,
            dtype=np.float32,
        )
    return simulate_ks(
        L=config.L,
        Q=config.Q,
        dt=config.dt,
        n_steps=config.total_steps - config.burn_in,
        mu=config.mu,
        lam=config.lam,
        seed=args.seed,
        burn_in=config.burn_in,
        dtype=np.float32,
    )


def build_model(args, device, dtype, seed):
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
        sigma=args.sigma,
        kappa=args.kappa,
    ).to(device=device, dtype=dtype)


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
    cfg = resolve_experiment_config(args, mode="eval")
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    resolved = config_to_namespace(cfg)
    data = load_or_simulate(args, resolved)

    u_train = torch.tensor(data[: resolved.T_train], dtype=dtype, device=device)
    u_eval = torch.tensor(
        data[resolved.T_train : resolved.T_train + resolved.T_eval],
        dtype=dtype,
        device=device,
    )

    rmse_trials = []
    for trial in range(resolved.n_trials):
        seed = args.seed + trial
        model = build_model(resolved, device, dtype, seed)
        rmse_trials.append(evaluate_trial(resolved, model, u_train, u_eval))

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
        "config": cfg,
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
