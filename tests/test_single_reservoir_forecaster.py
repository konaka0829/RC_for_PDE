import torch

from torchesn.nn.single_reservoir_forecaster import SingleReservoirForecaster


def test_single_reservoir_forecaster_train_predict():
    torch.manual_seed(0)
    t = torch.linspace(0, 4 * torch.pi, 120)
    series = torch.stack([torch.sin(t), torch.cos(t)], dim=1)

    model = SingleReservoirForecaster(
        input_dim=2,
        reservoir_size=60,
        degree=4,
        spectral_radius=0.9,
        sigma=1.0,
        beta=1e-3,
        seed=0,
    )

    model.fit(series[:100], washout=5)
    model.sync(series[100:110])
    preds = model.predict(series[109], steps=10)

    assert preds.shape == (10, 2)
    assert torch.isfinite(preds).all()
