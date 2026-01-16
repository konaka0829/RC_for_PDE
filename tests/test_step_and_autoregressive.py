import torch

from torchesn.nn import ESN


def test_step_matches_single_timestep_forward():
    torch.manual_seed(0)
    batch_size = 3
    input_size = 4
    hidden_size = 6
    output_size = 5

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        readout_training="gd",
    )

    inputs = torch.randn(1, batch_size, input_size)
    washout = [0] * batch_size

    forward_output, forward_hidden = model(inputs, washout)
    step_output, step_hidden = model.step(inputs[0], washout=washout)

    torch.testing.assert_close(forward_output[0], step_output)
    torch.testing.assert_close(forward_hidden, step_hidden)


def test_predict_autoregressive_output_shape():
    torch.manual_seed(1)
    batch_size = 2
    input_size = 3
    hidden_size = 4
    output_size = input_size
    steps = 4

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        readout_training="gd",
    )

    init_u = torch.randn(batch_size, input_size)
    outputs = model.predict_autoregressive(init_u, steps)

    assert outputs.shape == (steps, batch_size, output_size)
