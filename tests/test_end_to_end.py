import numpy as np
import torch

from ks_basic_single_reservoir_torch import (
    ModelParams,
    ReservoirParams,
    kursiv_solve,
    predict,
    train_reservoir,
)


def test_end_to_end_pipeline():
    model_params = ModelParams(tau=0.25, nstep=300, N=16, d=22.0)
    rng = np.random.default_rng(123)
    init_cond = 0.6 * (-1.0 + 2.0 * rng.random(model_params.N))

    uu = kursiv_solve(init_cond, model_params)
    data = uu.T

    measurements = data
    num_inputs = measurements.shape[0]

    resparams = ReservoirParams(
        radius=0.6,
        degree=3.0,
        N=(160 // num_inputs) * num_inputs,
        sigma=0.5,
        train_length=200,
        num_inputs=num_inputs,
        predict_length=20,
        beta=1e-4,
    )

    measurements_tensor = torch.as_tensor(measurements, dtype=torch.float64)
    train_data = measurements_tensor[:, : resparams.train_length]
    x, wout, A_sparse, win = train_reservoir(resparams, train_data)
    output, _ = predict(A_sparse, win, resparams, x, wout)

    assert output.shape == (num_inputs, resparams.predict_length)
    assert torch.isfinite(output).all()
