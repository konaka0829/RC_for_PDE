"""Single-reservoir ESN demo for Kuramoto–Sivashinsky forecasting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.nn.echo_state_network import ESN
from torchesn.pde import simulate_ks_etdrk4
from torchesn.utils import prepare_target


@dataclass
class KSRunConfig:
    n_grid: int = 64
    dt: float = 0.25
    domain_length: float = 22.0
    mu: float = 0.0
    train_steps: int = 2000
    pred_steps: int = 500
    warmup_steps: int = 200
    hidden_size: int = 500
    spectral_radius: float = 0.9
    leaking_rate: float = 1.0
    lambda_reg: float = 1e-4
    seed: int = 0
    output_dir: Path = Path("outputs")
    save_plots: bool = True


def generate_ks_data(config: KSRunConfig) -> np.ndarray:
    total_steps = config.train_steps + config.pred_steps + 1
    result = simulate_ks_etdrk4(
        n_steps=total_steps,
        dt=config.dt,
        n_grid=config.n_grid,
        domain_length=config.domain_length,
        mu=config.mu,
        seed=config.seed,
    )
    return result["u"]


def train_esn(
    train_u: torch.Tensor,
    *,
    hidden_size: int,
    spectral_radius: float,
    leaking_rate: float,
    lambda_reg: float,
) -> ESN:
    model = ESN(
        input_size=train_u.size(-1),
        hidden_size=hidden_size,
        output_size=train_u.size(-1),
        spectral_radius=spectral_radius,
        leaking_rate=leaking_rate,
        lambda_reg=lambda_reg,
        readout_training="cholesky",
        output_steps="all",
        feature_transform="square_even",
    )
    seq_len = train_u.size(0)
    target_seq = train_u[1:]
    input_seq = train_u[:-1]
    seq_lengths = [seq_len - 1]
    washout = [0]
    flat_target = prepare_target(target_seq, seq_lengths, washout)
    model(input_seq, washout, target=flat_target)
    model.fit()
    return model


def autoregressive_rollout(
    model: ESN,
    warmup_u: torch.Tensor,
    pred_steps: int,
) -> torch.Tensor:
    with torch.no_grad():
        _, hidden = model(warmup_u, washout=[0])
        current = warmup_u[-1:, :, :]
        predictions = []
        for _ in range(pred_steps):
            output, hidden = model(current, washout=[0], h_0=hidden)
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite values detected during rollout.")
            predictions.append(output[0, 0])
            current = output
        return torch.stack(predictions, dim=0)


def compute_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    error = prediction - target
    rmse = torch.sqrt(torch.mean(error ** 2))
    if not torch.isfinite(rmse):
        raise RuntimeError("RMSE is not finite.")
    return float(rmse)


def save_rmse_plot(rmse_series: torch.Tensor, output_dir: Path) -> None:
    rmse_np = rmse_series.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rmse_np)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("RMSE")
    ax.set_title("KS single-reservoir RMSE")
    ax.grid(True, alpha=0.3)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"ks_single_reservoir_rmse.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_prediction_heatmap(prediction: torch.Tensor, output_dir: Path) -> None:
    pred_np = prediction.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    heatmap = ax.imshow(
        pred_np,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_xlabel("Spatial index")
    ax.set_ylabel("Prediction step")
    ax.set_title("KS single-reservoir prediction (heatmap)")
    fig.colorbar(heatmap, ax=ax, label="u")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"ks_single_reservoir_prediction_heatmap.{ext}",
            bbox_inches="tight",
        )
    plt.close(fig)


def run_experiment(config: KSRunConfig) -> float:
    u = generate_ks_data(config)
    train_u = torch.tensor(
        u[: config.train_steps + 1], dtype=torch.float32
    ).unsqueeze(1)
    warmup_start = max(0, config.train_steps - config.warmup_steps)
    warmup_u = torch.tensor(
        u[warmup_start: config.train_steps + 1], dtype=torch.float32
    ).unsqueeze(1)
    target_u = torch.tensor(
        u[config.train_steps + 1: config.train_steps + 1 + config.pred_steps],
        dtype=torch.float32,
    )

    model = train_esn(
        train_u,
        hidden_size=config.hidden_size,
        spectral_radius=config.spectral_radius,
        leaking_rate=config.leaking_rate,
        lambda_reg=config.lambda_reg,
    )
    prediction = autoregressive_rollout(model, warmup_u, config.pred_steps)
    rmse = compute_rmse(prediction, target_u)
    if config.save_plots:
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rmse_series = torch.sqrt(
            torch.mean((prediction - target_u) ** 2, dim=1)
        )
        save_rmse_plot(rmse_series, output_dir)
        save_prediction_heatmap(prediction, output_dir)
    return rmse


def parse_args() -> KSRunConfig:
    parser = argparse.ArgumentParser(
        description="Single-reservoir ESN KS forecasting demo."
    )
    parser.add_argument("--n-grid", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--domain-length", type=float, default=22.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--pred-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=500)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--leaking-rate", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    return KSRunConfig(
        n_grid=args.n_grid,
        dt=args.dt,
        domain_length=args.domain_length,
        mu=args.mu,
        train_steps=args.train_steps,
        pred_steps=args.pred_steps,
        warmup_steps=args.warmup_steps,
        hidden_size=args.hidden_size,
        spectral_radius=args.spectral_radius,
        leaking_rate=args.leaking_rate,
        lambda_reg=args.lambda_reg,
        seed=args.seed,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
    )


def main() -> None:
    config = parse_args()
    rmse = run_experiment(config)
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
