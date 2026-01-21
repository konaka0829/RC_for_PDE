import torch

from torchesn.ks.benchmark import rmse_over_space
from torchesn.ks.ks_parallel_reservoir import KSParallelParams, KSParallelReservoir
from torchesn.ks.solver import generate_ks_dataset, ks_solve_etdrk4_forced


def test_forced_solver_and_dataset_shapes() -> None:
    init = torch.zeros(8, dtype=torch.float64)
    uu = ks_solve_etdrk4_forced(init, dt=0.25, n_steps=5, L=22.0, mu=0.0, wavelength=11.0)
    assert uu.shape == (5, 8)

    data = generate_ks_dataset(Q=8, L=22.0, mu=0.0, wavelength=11.0, dt=0.25, n_steps=6, burn_in=2, seed=0)
    assert data.shape == (8, 6)


def test_parallel_reservoir_fit_predict() -> None:
    params = KSParallelParams(
        Q=4,
        g=2,
        locality=1,
        approx_reservoir_size=12,
        radius=0.5,
        degree=3.0,
        beta=1e-3,
        sigma=0.5,
        discard_length=1,
        train_length=4,
        predict_length=3,
        jobid=1,
    )
    model = KSParallelReservoir(params, device="cpu", dtype=torch.float64, power_iter=20)
    u = torch.randn(4, 5, dtype=torch.float64)
    model.fit(u, time_chunk=2)
    pred = model.predict_one_interval(u, warmup_start=1, sync_length=1, predict_length=2)
    assert pred.shape == (4, 2)

    rmse = rmse_over_space(u[:, 1:3] * params.sigma, pred)
    assert rmse.shape == (2,)
