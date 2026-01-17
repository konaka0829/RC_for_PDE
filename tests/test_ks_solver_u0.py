from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.utils import simulate_ks


def test_simulate_ks_uses_u0_and_shapes():
    Q = 8
    L = 10.0
    dt = 0.01
    mu = 0.0
    lam = 0.0
    n_steps = 3

    u0 = torch.linspace(0.0, 1.0, Q)
    traj = simulate_ks(L, Q, dt, mu, lam, n_steps, u0=u0)

    assert traj.shape == (n_steps + 1, Q)
    assert torch.allclose(traj[0], u0)

    traj_random = simulate_ks(L, Q, dt, mu, lam, n_steps, seed=0)
    assert traj_random.shape == (n_steps + 1, Q)
    assert not torch.allclose(traj_random[0], u0)
