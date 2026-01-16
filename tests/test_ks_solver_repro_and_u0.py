import numpy as np

from torchesn.utils.kuramoto_sivashinsky import simulate_ks


def test_ks_solver_reproducible():
    data_a = simulate_ks(
        L=22.0,
        Q=32,
        dt=0.25,
        n_steps=20,
        mu=0.0,
        seed=123,
        dtype=np.float32,
    )
    data_b = simulate_ks(
        L=22.0,
        Q=32,
        dt=0.25,
        n_steps=20,
        mu=0.0,
        seed=123,
        dtype=np.float32,
    )

    assert np.array_equal(data_a, data_b)


def test_ks_solver_u0_is_first_state():
    Q = 16
    u0 = np.linspace(-1.0, 1.0, Q, dtype=np.float32)
    data = simulate_ks(
        L=22.0,
        Q=Q,
        dt=0.25,
        n_steps=5,
        mu=0.0,
        u0=u0,
        dtype=np.float32,
    )

    assert np.allclose(data[0], u0)
