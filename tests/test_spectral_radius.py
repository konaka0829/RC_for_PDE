from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.utils import estimate_spectral_radius_power_iteration


def test_estimate_spectral_radius_small_matches_target():
    torch.manual_seed(0)
    size = 20
    target = 0.9
    W = torch.randn(size, size)

    estimate = estimate_spectral_radius_power_iteration(W, n_iter=100)
    scaled = W * (target / estimate)
    true_radius = torch.max(torch.abs(torch.linalg.eigvals(scaled)))

    assert torch.isfinite(true_radius)
    assert torch.isclose(true_radius.real, torch.tensor(target), rtol=0.2, atol=0.05)


def test_estimate_spectral_radius_large_is_finite():
    torch.manual_seed(0)
    size = 800
    W = torch.randn(size, size)

    estimate = estimate_spectral_radius_power_iteration(W, n_iter=20)

    assert torch.isfinite(estimate)
    assert estimate > 0
