import pytest
import torch

from torchesn.nn import Reservoir
from torchesn.utils import estimate_spectral_radius_power_iteration


def test_spectral_radius_small_hidden_size():
    torch.manual_seed(0)

    input_size = 3
    hidden_size = 32
    spectral_radius = 0.9
    reservoir = Reservoir(
        mode="RES_TANH",
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        leaking_rate=1.0,
        spectral_radius=spectral_radius,
        w_ih_scale=torch.ones(input_size + 1),
        density=0.5,
    )

    w_hh = reservoir.weight_hh_l0.detach()
    abs_eigs = torch.abs(torch.linalg.eigvals(w_hh))
    max_abs = torch.max(abs_eigs).item()
    assert max_abs == pytest.approx(spectral_radius, rel=0.15, abs=0.15)


def test_spectral_radius_large_hidden_size_uses_power_iteration(monkeypatch):
    torch.manual_seed(0)

    def _raise(*args, **kwargs):
        raise RuntimeError("eigvals should not be called for large reservoirs")

    monkeypatch.setattr(torch.linalg, "eigvals", _raise)

    input_size = 3
    hidden_size = 300
    spectral_radius = 0.95
    reservoir = Reservoir(
        mode="RES_TANH",
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        leaking_rate=1.0,
        spectral_radius=spectral_radius,
        w_ih_scale=torch.ones(input_size + 1),
        density=0.1,
    )

    w_hh = reservoir.weight_hh_l0.detach()
    estimate = estimate_spectral_radius_power_iteration(w_hh)
    assert torch.isfinite(torch.tensor(estimate))
    assert estimate > 0
