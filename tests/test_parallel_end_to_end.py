import torch

from torchesn.nn.parallel_echo_state_network import ParallelESN


def _make_toy_data(T, Q):
    u = torch.zeros(T, Q)
    u[0] = torch.linspace(0.0, 1.0, Q)
    for t in range(T - 1):
        u[t + 1] = 0.9 * u[t] + 0.1 * torch.roll(u[t], shifts=1, dims=0)
    return u


def test_parallel_fit_predict_runs():
    torch.manual_seed(0)

    T = 30
    Q = 6
    u_train = _make_toy_data(T, Q)

    model = ParallelESN(
        Q=Q,
        g=2,
        l=1,
        hidden_size=12,
        spectral_radius=0.9,
        leaking_rate=1.0,
        density=0.5,
        lambda_reg=1e-4,
        readout_training="cholesky",
    )

    model.fit(u_train, washout=0)
    pred = model.predict(u_train[-5:], steps=4, epsilon=2)

    assert pred.shape == (4, Q)
    assert torch.isfinite(pred).all()


def test_parallel_matches_esn_when_single_segment():
    torch.manual_seed(0)

    Q = 5
    steps = 4
    u_hist = torch.randn(2, Q)

    parallel = ParallelESN(
        Q=Q,
        g=1,
        l=0,
        hidden_size=6,
        readout_training="gd",
        readout_features="linear",
        translation_invariant=True,
    )

    esn = parallel.esn_shared

    parallel_pred = parallel.predict(u_hist, steps=steps, epsilon=0)
    esn_pred = esn.predict_autoregressive(u_hist[-1], steps=steps).squeeze(1)

    assert torch.allclose(parallel_pred, esn_pred, atol=1e-6, rtol=1e-5)


def test_parallel_matches_esn_with_warmup():
    torch.manual_seed(0)

    Q = 5
    epsilon = 3
    steps = 4
    u_hist = torch.randn(epsilon + 1, Q)

    parallel = ParallelESN(
        Q=Q,
        g=1,
        l=0,
        hidden_size=6,
        readout_training="gd",
        readout_features="linear",
        translation_invariant=True,
    )
    esn = parallel.esn_shared

    hx = None
    for t in range(epsilon):
        _, hx = esn.step(u_hist[t].unsqueeze(0), hx=hx)

    parallel_pred = parallel.predict(u_hist, steps=steps, epsilon=epsilon)
    esn_pred = esn.predict_autoregressive(u_hist[-1], steps=steps, hx=hx).squeeze(1)

    assert torch.allclose(parallel_pred, esn_pred, atol=1e-6, rtol=1e-5)
