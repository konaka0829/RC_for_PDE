import torch

from torchesn.utils import simulate_ks


def test_simulate_ks_uses_u0_and_shapes():
    Q = 5
    n_steps = 3
    u0 = torch.linspace(0, 1, Q)

    out_with_u0 = simulate_ks(
        L=10.0,
        Q=Q,
        dt=0.1,
        mu=1.0,
        lam=0.5,
        n_steps=n_steps,
        u0=u0,
        seed=123,
    )
    assert out_with_u0.shape == (n_steps + 1, Q)
    torch.testing.assert_close(out_with_u0[0], u0)

    out_without_u0 = simulate_ks(
        L=10.0,
        Q=Q,
        dt=0.1,
        mu=1.0,
        lam=0.5,
        n_steps=n_steps,
        u0=None,
        seed=123,
    )
    assert out_without_u0.shape == (n_steps + 1, Q)
    assert not torch.allclose(out_without_u0[0], u0)
