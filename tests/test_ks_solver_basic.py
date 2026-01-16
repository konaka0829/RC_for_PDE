import numpy as np

from torchesn.utils.kuramoto_sivashinsky import simulate_ks


def test_ks_solver_basic():
    data = simulate_ks(
        L=22.0,
        Q=32,
        dt=0.25,
        n_steps=40,
        burn_in=10,
        mu=0.0,
        seed=0,
        dtype=np.float32,
    )

    assert data.shape == (40, 32)
    assert np.isfinite(data).all()
