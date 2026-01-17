import torch

from torchesn.nn.sparse_reservoir import generate_sparse_reservoir, _estimate_spectral_radius


def test_sparse_reservoir_spectral_radius():
    size = 50
    degree = 4
    target = 0.9
    A = generate_sparse_reservoir(size=size, degree=degree, spectral_radius=target, seed=0)
    estimate = _estimate_spectral_radius(A, power_iters=120)

    assert estimate > 0
    assert abs(estimate - target) / target < 0.2


def test_sparse_reservoir_step_finite():
    size = 40
    degree = 3
    A = generate_sparse_reservoir(size=size, degree=degree, spectral_radius=0.8, seed=1)
    x = torch.randn(size)
    u = torch.randn(size)
    out = torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1) + u
    assert torch.isfinite(out).all()
