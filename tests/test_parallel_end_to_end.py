from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.nn import ParallelESN


def test_parallel_esn_end_to_end():
    torch.manual_seed(0)
    Q = 6
    g = 2
    l = 1
    hidden_size = 8
    steps = 3

    model = ParallelESN(
        Q=Q,
        g=g,
        l=l,
        mu=0.0,
        hidden_size=hidden_size,
        spectral_radius=0.9,
        leaking_rate=1.0,
        density=1.0,
        lambda_reg=1e-3,
        readout_training='cholesky',
        readout_features='linear',
        w_io=False,
        translation_invariant=True,
        seed=0,
    )

    u_train = torch.randn(6, 1, Q)
    model.fit(u_train, washout=0)

    u_hist = u_train[:3]
    assert model.shared_esn is not None
    assert model._get_esn(0) is model._get_esn(1)

    outputs = model.predict(u_hist, steps=steps)

    assert outputs.shape == (steps, 1, Q)
    assert torch.isfinite(outputs).all()
