"""End-to-end demo mirroring the MATLAB ks.m script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ks_basic_single_reservoir_torch import (
    ModelParams,
    ReservoirParams,
    kursiv_solve,
    predict,
    train_reservoir,
)


def main() -> None:
    model_params = ModelParams(tau=0.25, nstep=100000, N=64, d=22.0)
    rng = np.random.default_rng()
    init_cond = 0.6 * (-1.0 + 2.0 * rng.random(model_params.N))

    uu = kursiv_solve(init_cond, model_params)
    data = uu.T

    measured_vars = np.arange(model_params.N)
    measurements = data[measured_vars, :]

    num_inputs = measurements.shape[0]
    approx_res_size = 3000
    resparams = ReservoirParams(
        radius=0.6,
        degree=3.0,
        N=(approx_res_size // num_inputs) * num_inputs,
        sigma=0.5,
        train_length=70000,
        num_inputs=num_inputs,
        predict_length=2000,
        beta=0.0001,
    )

    measurements_tensor = torch.as_tensor(measurements, dtype=torch.float64)
    train_data = measurements_tensor[:, : resparams.train_length]
    x, wout, A_sparse, win = train_reservoir(resparams, train_data)
    output, _ = predict(A_sparse, win, resparams, x, wout)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
