import torch

from torchesn.nn.parallel_reservoir_forecaster import ParallelReservoirForecaster
from torchesn.nn.single_reservoir_forecaster import SingleReservoirForecaster


def _make_series(T: int, Q: int) -> torch.Tensor:
    t = torch.linspace(0, 2 * torch.pi, T)
    base = torch.stack([torch.sin(t), torch.cos(t)], dim=1)
    reps = (Q + base.shape[1] - 1) // base.shape[1]
    series = base.repeat(1, reps)[:, :Q]
    return series


def test_parallel_forecaster_fit_predict():
    Q = 16
    g = 4
    l = 1
    train = _make_series(500, Q)
    test = _make_series(120, Q)

    model = ParallelReservoirForecaster(
        Q=Q,
        g=g,
        l=l,
        reservoir_size_approx=200,
        degree=3,
        spectral_radius=0.6,
        beta=1e-3,
        seed=0,
    )

    model.fit(train, washout=10)
    preds = model.predict(test[:20], predict_length=30)

    assert preds.shape == (30, Q)
    assert torch.isfinite(preds).all()


def test_parallel_forecaster_predict_before_fit():
    Q = 8
    model = ParallelReservoirForecaster(
        Q=Q,
        g=2,
        l=1,
        reservoir_size_approx=50,
        degree=3,
        spectral_radius=0.6,
        beta=1e-3,
        seed=0,
    )
    sync = _make_series(10, Q)

    try:
        model.predict(sync, predict_length=5)
    except ValueError as exc:
        assert "Model has not been fit yet" in str(exc)
    else:
        raise AssertionError("predict should fail before fit")


def test_parallel_matches_single_shape():
    Q = 6
    train = _make_series(120, Q)
    sync = _make_series(20, Q)

    parallel = ParallelReservoirForecaster(
        Q=Q,
        g=1,
        l=0,
        reservoir_size_approx=60,
        degree=3,
        spectral_radius=0.7,
        beta=1e-3,
        seed=1,
    )
    parallel.fit(train, washout=5)
    pred_parallel = parallel.predict(sync[:10], predict_length=5)

    single = SingleReservoirForecaster(
        input_dim=Q,
        reservoir_size=60,
        degree=3,
        spectral_radius=0.7,
        beta=1e-3,
        seed=1,
    )
    single.fit(train, washout=5)
    single.sync(sync[:10])
    pred_single = single.predict(sync[9], steps=5)

    assert pred_parallel.shape == pred_single.shape
    assert torch.isfinite(pred_parallel).all()
    assert torch.isfinite(pred_single).all()
