import torch

from ks_basic_single_reservoir_torch.ks_solver import ModelParams, kursiv_solve


def test_kursiv_solve_output_shape():
    params = ModelParams(N=32, d=22.0, tau=0.25, nstep=5)
    init = torch.randn(params.N)
    out = kursiv_solve(init, params, device=torch.device("cpu"))
    assert out.shape == (params.nstep, params.N)


def test_kursiv_solve_zero_init_is_zero():
    params = ModelParams(N=32, d=22.0, tau=0.25, nstep=4)
    init = torch.zeros(params.N)
    out = kursiv_solve(init, params, device=torch.device("cpu"))
    assert torch.allclose(out, torch.zeros_like(out))


def test_kursiv_solve_no_nan_inf():
    params = ModelParams(N=32, d=22.0, tau=0.25, nstep=3)
    init = torch.randn(params.N)
    out = kursiv_solve(init, params, device=torch.device("cpu"))
    assert torch.isfinite(out).all()
