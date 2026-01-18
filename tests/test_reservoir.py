import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from ks_basic_single_reservoir_torch.reservoir import (
    ReservoirParams,
    build_win,
    generate_reservoir_scipy,
    predict,
    reservoir_layer,
    scipy_sparse_to_torch,
    train_wout,
)


def _default_params(**overrides):
    params = ReservoirParams(
        N=12,
        radius=0.9,
        degree=3.0,
        sigma=0.5,
        train_length=5,
        num_inputs=3,
        predict_length=2,
        beta=1e-4,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_build_win_block_structure_and_seed_reset():
    params = _default_params(N=12, num_inputs=3, sigma=0.5)
    win = build_win(params, device=torch.device("cpu"), dtype=torch.float64)
    q = params.N // params.num_inputs

    for i in range(1, params.num_inputs + 1):
        rng = np.random.RandomState(i)
        ip = params.sigma * (-1.0 + 2.0 * rng.rand(q, 1))
        expected = torch.zeros(params.N, dtype=torch.float64)
        expected[(i - 1) * q : i * q] = torch.from_numpy(ip[:, 0])
        assert torch.allclose(win[:, i - 1], expected)

    for col in range(params.num_inputs):
        nonzero_indices = torch.nonzero(win[:, col], as_tuple=False).squeeze(1)
        assert len(nonzero_indices) == q
        assert nonzero_indices.min().item() == col * q
        assert nonzero_indices.max().item() == (col + 1) * q - 1


def test_reservoir_layer_ignores_last_input_column():
    params = _default_params(train_length=4, num_inputs=2, N=4)
    device = torch.device("cpu")
    A_csr = scipy.sparse.identity(params.N, format="csr")
    A_sparse = scipy_sparse_to_torch(A_csr, device=device, dtype=torch.float64)
    win = torch.ones((params.N, params.num_inputs), dtype=torch.float64)

    data1 = torch.randn((params.num_inputs, params.train_length), dtype=torch.float64)
    data2 = data1.clone()
    data2[:, -1] = data2[:, -1] + 1000.0

    states1 = reservoir_layer(A_sparse, win, data1, params)
    states2 = reservoir_layer(A_sparse, win, data2, params)

    assert torch.allclose(states1, states2)


def test_train_wout_does_not_mutate_states():
    params = _default_params(N=6, train_length=4, num_inputs=2)
    states = torch.randn((params.N, params.train_length), dtype=torch.float64)
    states_before = states.clone()
    data = torch.randn((params.num_inputs, params.train_length), dtype=torch.float64)

    _ = train_wout(params, states, data)

    assert torch.allclose(states, states_before)


def test_predict_uses_unsquared_state_for_update():
    params = _default_params(N=4, num_inputs=2, predict_length=1)
    device = torch.device("cpu")
    A_csr = scipy.sparse.identity(params.N, format="csr")
    A_sparse = scipy_sparse_to_torch(A_csr, device=device, dtype=torch.float64)
    win = torch.zeros((params.N, params.num_inputs), dtype=torch.float64)
    wout = torch.zeros((params.num_inputs, params.N), dtype=torch.float64)
    x0 = torch.tensor([0.2, -0.4, 0.6, -0.8], dtype=torch.float64)

    _, x_last = predict(A_sparse, win, params, x0, wout)

    expected = torch.tanh(x0)
    assert torch.allclose(x_last, expected)


def test_generate_reservoir_density():
    size = 50
    degree = 5.0
    radius = 0.9
    A = generate_reservoir_scipy(size, radius, degree, seed=123)
    density = A.nnz / (size * size)
    expected = degree / size
    assert abs(density - expected) < 0.03


def test_generate_reservoir_scaling():
    size = 30
    degree = 4.0
    radius = 0.8
    A = generate_reservoir_scipy(size, radius, degree, seed=456)
    eigvals = scipy.sparse.linalg.eigs(A, k=min(6, size - 2), which="LM", return_eigenvectors=False)
    max_abs = np.max(np.abs(eigvals))
    assert np.isclose(max_abs, radius, atol=1e-3)
