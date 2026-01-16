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
from torchesn.utils.kuramoto_sivashinsky import simulate_ks

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def apply_paper_defaults(args):
    args.hidden_size = 5000
    args.T_train = 70000
    args.spectral_radius = 0.6
    args.sigma = 1.0
    args.l = 6
    args.kappa = 3
    args.K = 30
    args.tau = 1000
    args.epsilon = 10
    args.n_trials = 10


def apply_quick_defaults(args):
    args.Q = 64
    args.g = 8
    args.l = 2
    args.hidden_size = 150
    args.T_train = 400
    args.T_eval = 200
    args.K = 2
    args.tau = 60
    args.epsilon = 10
    args.n_trials = 1
    args.lambda_reg = max(args.lambda_reg, 1e-2)
    if args.Q % args.g != 0:
        args.Q = args.g * (args.Q // args.g)


def build_model(args, mu, translation_invariant=False):
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
        translation_invariant=translation_invariant,
        mu=mu,
        seed=args.seed,
    )


def generate_ks(args, mu, total_steps):
    return simulate_ks(
        L=args.L,
        Q=args.Q,
        dt=args.dt,
        n_steps=total_steps,
        mu=mu,
        lam=args.lam,
        seed=args.seed,
        burn_in=args.burn_in,
        dtype=np.float32,
    )


def train_model(args, mu):
    total_steps = args.T_train + args.T_eval + args.epsilon + args.tau + 1
    data = generate_ks(args, mu, total_steps=total_steps)
    u_train = torch.tensor(data[: args.T_train], dtype=torch.float32)
    u_eval = torch.tensor(data[args.T_train :], dtype=torch.float32)
    model = build_model(args, mu)
    model.fit(u_train, washout=0)
    return model, u_eval


def rmse_curve(model, u_eval, args):
    rmse = model.evaluate_windows(u_eval, args.K, args.tau, args.epsilon)
    rmse = rmse.detach().cpu().numpy()
    print(f"RMSE curve (mean): {rmse.mean():.6f}")
    return rmse


def plot_space_time(data, path, title):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(data, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fig2_single(args):
    args.g = 1
    args.l = 0
    if args.quick:
        apply_quick_defaults(args)
    model, u_eval = train_model(args, mu=args.mu)
    u_hist = u_eval[: args.epsilon + 1]
    pred = model.predict(u_hist, steps=args.tau, epsilon=args.epsilon)
    true = u_eval[args.epsilon + 1 : args.epsilon + 1 + args.tau]
    error = true - pred
    rmse = torch.sqrt(torch.mean(error ** 2)).item()
    print(f"RMSE window: {rmse:.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_space_time(true, out_dir / "fig2_true.png", "Fig2 true")
    plot_space_time(pred, out_dir / "fig2_pred.png", "Fig2 pred")
    plot_space_time(error, out_dir / "fig2_error.png", "Fig2 error")


def fig4_parallel(args):
    if args.quick:
        apply_quick_defaults(args)
        args.mu = 0.01
        args.lam = 100.0
        args.L = 44.0
    model, u_eval = train_model(args, mu=args.mu)
    u_hist = u_eval[: args.epsilon + 1]
    pred = model.predict(u_hist, steps=args.tau, epsilon=args.epsilon)
    true = u_eval[args.epsilon + 1 : args.epsilon + 1 + args.tau]
    error = true - pred
    rmse = torch.sqrt(torch.mean(error ** 2)).item()
    print(f"RMSE window: {rmse:.6f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_space_time(true, out_dir / "fig4_true.png", "Fig4 true")
    plot_space_time(pred, out_dir / "fig4_pred.png", "Fig4 pred")
    plot_space_time(error, out_dir / "fig4_error.png", "Fig4 error")

    u_hat0 = pred[0].detach().cpu().numpy()
    ks_from_pred = simulate_ks(
        L=args.L,
        Q=args.Q,
        dt=args.dt,
        n_steps=args.tau,
        mu=args.mu,
        lam=args.lam,
        u0=u_hat0,
        burn_in=0,
        dtype=np.float32,
    )
    error_ks = true.detach().cpu().numpy() - ks_from_pred
    plot_space_time(error_ks, out_dir / "fig4_error_ks.png", "Fig4 error KS")
    print("Fig4: using u_hat0 from prediction for KS restart.")


def fig5_scaling(args):
    if args.quick:
        apply_quick_defaults(args)
        args.L = 200.0
        args.Q = 64
        args.mu = 0.01
        args.lam = 100.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    L_values = [200, 400, 800, 1600]
    g_values = [64, 128, 256, 512]
    if args.quick:
        L_values = [100, 200]
        g_values = [8, 16]

    dx_base = 200.0 / 512.0
    curves = []
    settings = []
    for L, g in zip(L_values, g_values):
        Q = int(round(L / dx_base))
        if Q % g != 0:
            Q = g * ((Q + g - 1) // g)
        print(f"Fig5a: L={L}, g={g}, Q={Q} (adjusted).")
        args.L = float(L)
        args.g = int(g)
        args.Q = int(Q)
        model, u_eval = train_model(args, mu=args.mu)
        curve = rmse_curve(model, u_eval, args)
        curves.append(curve)
        settings.append({"L": L, "g": g, "Q": Q})

    plt.figure(figsize=(6, 4))
    for curve, setting in zip(curves, settings):
        plt.plot(curve, label=f"L={setting['L']}, g={setting['g']}")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_a.png")
    plt.close()

    args.L = 200.0
    args.Q = 512 if not args.quick else 64
    args.g = 64 if not args.quick else 8
    g_values = [1, 2, 4, 8, 16, 32, 64] if not args.quick else [1, 2, 4, 8]
    curves = []
    settings = []
    for g in g_values:
        if args.Q % g != 0:
            continue
        args.g = g
        model, u_eval = train_model(args, mu=args.mu)
        curve = rmse_curve(model, u_eval, args)
        curves.append(curve)
        settings.append({"g": g})

    plt.figure(figsize=(6, 4))
    for curve, setting in zip(curves, settings):
        plt.plot(curve, label=f"g={setting['g']}")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_b.png")
    plt.close()

    with open(out_dir / "fig5_settings.json", "w", encoding="utf-8") as f:
        json.dump({"scaling": settings}, f, indent=2)


def fig6_shared_weights(args):
    if args.quick:
        apply_quick_defaults(args)
        args.L = 44.0
        args.Q = 64

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    for mu in [0.0, 0.01]:
        model = build_model(args, mu=mu, translation_invariant=False)
        data = generate_ks(args, mu=mu, total_steps=args.T_train + args.T_eval + 1)
        u_train = torch.tensor(data[: args.T_train], dtype=torch.float32)
        u_eval = torch.tensor(data[args.T_train :], dtype=torch.float32)
        model.fit(u_train, washout=0)
        curves[f"independent_mu_{mu}"] = rmse_curve(model, u_eval, args)

        if mu != 0.0:
            print("Fig6: translation_invariant disabled for mu!=0 (independent only).")
            continue

        shared = build_model(args, mu=mu, translation_invariant=True)
        shared.fit(u_train, washout=0)
        curves[f"shared_mu_{mu}"] = rmse_curve(shared, u_eval, args)

    plt.figure(figsize=(6, 4))
    for key, curve in curves.items():
        plt.plot(curve, label=key)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(out_dir / "fig6.png")
    plt.close()


def add_common_args(parser):
    parser.add_argument("--L", type=float, default=22.0)
    parser.add_argument("--Q", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=100.0)
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--g", type=int, default=1)
    parser.add_argument("--l", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=500)
    parser.add_argument("--spectral-radius", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--kappa", type=int, default=3)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--leaking-rate", type=float, default=1.0)

    parser.add_argument("--readout-training", type=str, default="cholesky")
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--readout-features", type=str, default="linear_and_square")

    parser.add_argument("--T-train", type=int, default=1000)
    parser.add_argument("--T-eval", type=int, default=500)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--tau", type=int, default=100)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=1)

    parser.add_argument("--out-dir", type=str, default="examples/figures/prl2018")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--paper-defaults", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="PRL 2018 figure reproduction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["fig2_single", "fig4_parallel", "fig5_scaling"]:
        sub = subparsers.add_parser(name)
        add_common_args(sub)

    fig6 = subparsers.add_parser(
        "fig6_shared_weights",
        description="Shared weights disabled automatically when mu != 0.",
    )
    add_common_args(fig6)

    args = parser.parse_args()
    if args.paper_defaults:
        apply_paper_defaults(args)
    if args.quick:
        apply_quick_defaults(args)

    if args.command == "fig2_single":
        fig2_single(args)
    elif args.command == "fig4_parallel":
        fig4_parallel(args)
    elif args.command == "fig5_scaling":
        fig5_scaling(args)
    elif args.command == "fig6_shared_weights":
        fig6_shared_weights(args)


if __name__ == "__main__":
    main()
