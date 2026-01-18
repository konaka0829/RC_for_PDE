import numpy as np
import torch

from ks_basic_single_reservoir_torch.reservoir import (
    ReservoirParams,
    build_win,
    generate_reservoir_scipy,
    reservoir_layer,
    scipy_sparse_to_torch,
    train_reservoir_streaming,
    train_wout,
)


def test_streaming_train_matches_full():
    params = ReservoirParams(
        N=40,
        radius=0.6,
        degree=3.0,
        sigma=0.5,
        train_length=50,
        num_inputs=4,
        predict_length=10,
        beta=1e-4,
    )
    rng = np.random.default_rng(123)
    data = torch.as_tensor(rng.standard_normal((params.num_inputs, params.train_length)), dtype=torch.float64)

    seed_A = 42
    A_csr = generate_reservoir_scipy(params.N, params.radius, params.degree, seed=seed_A)
    A_sparse = scipy_sparse_to_torch(A_csr, device=torch.device("cpu"), dtype=torch.float64)
    win = build_win(params, device=torch.device("cpu"), dtype=torch.float64)

    states = reservoir_layer(A_sparse, win, data, params)
    wout_full = train_wout(params, states, data, method="pinv")
    x_last_full = states[:, -1]

    x_last_stream, wout_stream, A2, win2 = train_reservoir_streaming(
        params, data, device=torch.device("cpu"), seed_A=seed_A, block_size=16
    )

    assert torch.allclose(wout_full, wout_stream, rtol=1e-6, atol=1e-6)
    assert torch.allclose(x_last_full, x_last_stream, rtol=1e-6, atol=1e-6)
    assert torch.allclose(A_sparse.to_dense(), A2.to_dense())
    assert torch.allclose(win, win2)


def test_streaming_train_block_size_stability():
    params = ReservoirParams(
        N=40,
        radius=0.6,
        degree=3.0,
        sigma=0.5,
        train_length=50,
        num_inputs=4,
        predict_length=10,
        beta=1e-4,
    )
    rng = np.random.default_rng(456)
    data = torch.as_tensor(rng.standard_normal((params.num_inputs, params.train_length)), dtype=torch.float64)

    seed_A = 123
    _, wout_7, _, _ = train_reservoir_streaming(
        params, data, device=torch.device("cpu"), seed_A=seed_A, block_size=7
    )
    _, wout_16, _, _ = train_reservoir_streaming(
        params, data, device=torch.device("cpu"), seed_A=seed_A, block_size=16
    )

    assert torch.allclose(wout_7, wout_16, rtol=1e-5, atol=1e-6)
