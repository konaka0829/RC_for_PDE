import torch

from torchesn.nn import ESN


def test_step_matches_forward_single_timestep():
    torch.manual_seed(0)

    seq_len = 1
    batch_size = 2
    input_size = 3
    hidden_size = 4
    output_size = 3

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        readout_training="gd",
    )

    x = torch.randn(seq_len, batch_size, input_size)
    washout = [0] * batch_size

    forward_output, forward_hidden = model(x, washout)
    step_output, step_hidden = model.step(x[0])

    assert torch.allclose(forward_output[0], step_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(forward_hidden, step_hidden, atol=1e-6, rtol=1e-5)


def test_predict_autoregressive_shapes_and_finite():
    torch.manual_seed(0)

    batch_size = 2
    input_size = 3
    hidden_size = 5
    output_size = 3
    steps = 5

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        readout_training="gd",
    )

    init_u = torch.randn(batch_size, input_size)
    output = model.predict_autoregressive(init_u, steps=steps)

    assert output.shape == (steps, batch_size, output_size)
    assert torch.isfinite(output).all()


def test_step_respects_no_grad_context():
    torch.manual_seed(0)

    model = ESN(
        input_size=2,
        hidden_size=4,
        output_size=2,
        readout_training="gd",
    )
    input_t = torch.randn(1, 2)

    with torch.no_grad():
        output, _ = model.step(input_t)

    assert not output.requires_grad
