import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path("examples/figures/prl2018")


def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def _plot_and_save(name, x, y_true, y_pred):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.plot(x, y_true, label="true")
    plt.plot(x, y_pred, label="pred", linestyle="--")
    plt.legend()
    out_path = FIGURE_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _run_figure(name, quick):
    n = 20 if quick else 200
    x = np.linspace(0, 2 * np.pi, n)
    y_true = np.sin(x)
    y_pred = np.sin(x + (0.1 if quick else 0.05))
    rmse = _rmse(y_true, y_pred)
    out_path = _plot_and_save(name, x, y_true, y_pred)
    print(f"{name} RMSE: {rmse:.6f}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate PRL2018 figure placeholders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["fig2_single", "fig4_parallel", "fig5_scaling", "fig6_shared_weights"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--quick", action="store_true", help="Run quick mode")

    args = parser.parse_args()
    _run_figure(args.command, args.quick)


if __name__ == "__main__":
    main()
