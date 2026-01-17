import numpy as np

from torchesn.pde.ks_etdrk4 import simulate_ks_etdrk4


def test_simulate_ks_shape_and_finite():
    data = simulate_ks_etdrk4(
        L=50.0,
        Q=32,
        dt=0.25,
        n_steps=40,
        mu=0.01,
        lambda_=25.0,
        seed=0,
        dtype=np.float32,
    )

    assert data.shape == (40, 32)
    assert np.isfinite(data).all()


def test_simulate_ks_dtype_control():
    data32 = simulate_ks_etdrk4(
        L=50.0,
        Q=32,
        dt=0.25,
        n_steps=10,
        mu=0.01,
        lambda_=25.0,
        seed=1,
        dtype=np.float32,
    )
    data64 = simulate_ks_etdrk4(
        L=50.0,
        Q=32,
        dt=0.25,
        n_steps=10,
        mu=0.01,
        lambda_=25.0,
        seed=1,
        dtype=np.float64,
    )

    assert data32.dtype == np.float32
    assert data64.dtype == np.float64
