import torch

from torchesn.nn import ParallelESN


def test_parallel_esn_fit_and_predict_runs():
    torch.manual_seed(0)
    Q = 6
    g = 3
    l = 1
    seq_len = 6

    model = ParallelESN(
        Q=Q,
        g=g,
        l=l,
        hidden_size=8,
        spectral_radius=0.9,
        leaking_rate=1.0,
        density=1.0,
        lambda_reg=1e-3,
        readout_training="cholesky",
        readout_features="linear",
        w_io=False,
        seed=0,
    )

    u_train = torch.randn(seq_len, Q)
    model.fit(u_train, washout=0)

    u_hist = u_train[-3:]
    outputs = model.predict(u_hist, steps=2, epsilon=0.0)

    assert outputs.shape == (2, Q)
    assert torch.isfinite(outputs).all()


def test_parallel_esn_shared_weights_branch_runs():
    torch.manual_seed(1)
    Q = 6
    g = 3
    l = 1
    seq_len = 5

    model = ParallelESN(
        Q=Q,
        g=g,
        l=l,
        hidden_size=6,
        spectral_radius=0.9,
        leaking_rate=1.0,
        density=1.0,
        lambda_reg=1e-3,
        readout_training="cholesky",
        readout_features="linear",
        w_io=False,
        translation_invariant=True,
        mu=0,
        seed=1,
    )

    assert model.translation_invariant
    assert model.shared_esn is not None

    u_train = torch.randn(seq_len, Q)
    model.fit(u_train, washout=0)

    u_hist = u_train[-2:]
    outputs = model.predict(u_hist, steps=2, epsilon=0.0)

    assert outputs.shape == (2, Q)
    assert torch.isfinite(outputs).all()
