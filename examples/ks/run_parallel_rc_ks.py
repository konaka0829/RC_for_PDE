"""Run parallel reservoir computing experiment on KS datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torchesn.nn.parallel_reservoir_forecaster import ParallelReservoirForecaster, mean_rmse_over_segments, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel RC on KS datasets.")
    parser.add_argument("--data", required=True, help="Path to KS .npz dataset")
    parser.add_argument("--g", type=int, default=4, help="Number of reservoir groups")
    parser.add_argument("--l", type=int, default=1, help="Overlap size on each side")
    parser.add_argument("--reservoir-size-approx", type=int, default=200, help="Approx reservoir size per group")
    parser.add_argument("--degree", type=int, default=3, help="Average out-degree")
    parser.add_argument("--spectral-radius", type=float, default=0.6, help="Spectral radius scaling")
    parser.add_argument("--sigma", type=float, default=1.0, help="Input weight scale")
    parser.add_argument("--beta", type=float, default=1e-4, help="Ridge regression beta")
    parser.add_argument("--train-discard", type=int, default=0, help="Discard steps from start of train data")
    parser.add_argument("--train-length", type=int, default=None, help="Length of training window")
    parser.add_argument("--predict-length", type=int, default=100, help="Prediction length per interval")
    parser.add_argument("--sync-length", type=int, default=10, help="Sync length before prediction")
    parser.add_argument("--num-intervals", type=int, default=3, help="Number of prediction intervals")
    parser.add_argument("--interval-stride", type=int, default=None, help="Stride between intervals")
    parser.add_argument("--share-weights", action="store_true", help="Share weights across reservoirs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device for torch tensors")
    parser.add_argument("--dtype", type=str, choices=("float32", "float64"), default="float32")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for ridge statistics")
    parser.add_argument("--plot-out", type=str, default=None, help="Optional path to save RMSE plot")
    parser.add_argument("--paper", action="store_true", help="Use paper settings preset")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow loading legacy datasets with pickled metadata",
    )
    return parser.parse_args()


def _coerce_meta(meta_json: np.ndarray | str) -> dict:
    if isinstance(meta_json, np.ndarray):
        meta_json = meta_json.item()
    return json.loads(meta_json)


def load_dataset(path: Path, allow_pickle: bool) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta_json = data.get("meta_json")
            meta = _coerce_meta(meta_json) if meta_json is not None else None
            return {"train_u": data["train_u"], "test_u": data["test_u"], "meta": meta}
    except ValueError as exc:
        if not allow_pickle:
            raise ValueError(
                "Failed to load dataset without pickle. "
                "If this is a legacy dataset, re-generate it or pass --allow-pickle."
            ) from exc

    with np.load(path, allow_pickle=True) as data:
        meta = data.get("meta")
        if isinstance(meta, np.ndarray):
            meta = meta.item()
        return {"train_u": data["train_u"], "test_u": data["test_u"], "meta": meta}


def prepare_train_sequence(train_u: np.ndarray, discard: int, length: int | None) -> np.ndarray:
    if discard < 0:
        raise ValueError("train_discard must be non-negative")
    if discard >= train_u.shape[0]:
        raise ValueError("train_discard exceeds training length")
    if length is None:
        return train_u[discard:]
    if length <= 0:
        raise ValueError("train_length must be positive")
    end = discard + length
    if end > train_u.shape[0]:
        raise ValueError("train_length exceeds available training data")
    return train_u[discard:end]


def run_experiment(args: argparse.Namespace) -> List[torch.Tensor]:
    data = load_dataset(Path(args.data), allow_pickle=args.allow_pickle)
    train_u = data["train_u"]
    test_u = data["test_u"]
    meta = data.get("meta")

    if args.paper:
        args.g = 64
        args.l = 6
        args.reservoir_size_approx = 5000
        args.degree = 3
        args.spectral_radius = 0.6
        args.sigma = 1.0
        args.beta = 1e-4
        args.predict_length = 1000
        args.sync_length = 10
        args.num_intervals = 30
        args.interval_stride = args.predict_length
        args.train_length = 70000
        args.train_discard = 0

    interval_stride = args.interval_stride or args.predict_length
    required_test = (args.num_intervals - 1) * interval_stride + (args.sync_length + args.predict_length)
    actual_test_len = test_u.shape[0]
    train_steps = train_u.shape[0]
    if isinstance(meta, dict) and meta.get("train_steps") is not None:
        train_steps = int(meta["train_steps"])
    if actual_test_len < required_test:
        raise ValueError(
            f"Test data too short: need at least {required_test} steps, got {actual_test_len}. "
            f"(K={args.num_intervals}, stride={interval_stride}, sync={args.sync_length}, "
            f"pred={args.predict_length}). Regenerate with total_steps >= {train_steps + required_test}."
        )

    train_seq = prepare_train_sequence(train_u, args.train_discard, args.train_length)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)

    model = ParallelReservoirForecaster(
        Q=train_seq.shape[1],
        g=args.g,
        l=args.l,
        reservoir_size_approx=args.reservoir_size_approx,
        degree=args.degree,
        spectral_radius=args.spectral_radius,
        sigma=args.sigma,
        beta=args.beta,
        seed=args.seed,
        share_weights=args.share_weights,
        device=device,
        dtype=dtype,
    )

    model.fit(torch.from_numpy(train_seq).to(device=device, dtype=dtype), washout=0, chunk_size=args.chunk_size)

    rmse_segments: List[torch.Tensor] = []
    test_tensor = torch.from_numpy(test_u).to(device=device, dtype=dtype)
    for k in range(args.num_intervals):
        start = k * interval_stride
        end = start + args.sync_length + args.predict_length
        sync_inputs = test_tensor[start : start + args.sync_length]
        true_segment = test_tensor[start + args.sync_length : end]
        pred = model.predict(sync_inputs, predict_length=args.predict_length)
        rmse_segments.append(rmse(true_segment, pred))

    return rmse_segments


def main() -> None:
    args = parse_args()
    rmse_segments = run_experiment(args)
    stacked = torch.stack(rmse_segments, dim=0)
    mean_curve = stacked.mean(dim=0).cpu().numpy()
    segment_means = mean_rmse_over_segments(torch.tensor(mean_curve), segments=min(5, mean_curve.shape[0]))

    print("Mean RMSE (first 10 steps):", np.array2string(mean_curve[:10], precision=4))
    print("Mean RMSE over segments:", segment_means.cpu().numpy())

    if args.plot_out:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit(f"matplotlib is required for plotting: {exc}")

        plt.figure(figsize=(6, 4))
        plt.plot(mean_curve)
        plt.xlabel("Prediction step")
        plt.ylabel("RMSE")
        plt.title("Parallel RC Mean RMSE")
        out_path = Path(args.plot_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.with_suffix("")
        plt.tight_layout()
        for suffix in (".png", ".pdf", ".svg"):
            plt.savefig(stem.with_suffix(suffix))
        plt.close()


if __name__ == "__main__":
    main()
