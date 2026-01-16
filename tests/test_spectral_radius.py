import torch

from torchesn.utils import estimate_spectral_radius_power_iteration


def test_power_iteration_matches_small_matrix_radius():
    torch.manual_seed(0)
    matrix = torch.randn(8, 8)
    matrix = (matrix + matrix.t()) / 2

    true_radius = torch.max(torch.abs(torch.linalg.eigvals(matrix))).real
    estimated_radius = estimate_spectral_radius_power_iteration(matrix, n_iter=100)

    assert torch.isfinite(estimated_radius)
    assert torch.allclose(estimated_radius, true_radius, rtol=1e-2, atol=1e-2)


def test_power_iteration_large_matrix_is_finite():
    torch.manual_seed(1)
    matrix = torch.randn(512, 512)

    estimated_radius = estimate_spectral_radius_power_iteration(matrix, n_iter=30)

    assert torch.isfinite(estimated_radius)
