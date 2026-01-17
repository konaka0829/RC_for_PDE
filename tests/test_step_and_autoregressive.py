from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.nn import ESN


def test_step_matches_forward_single_timestep():
    torch.manual_seed(0)
    batch_size = 2
    input_size = 3
    output_size = 3

    model = ESN(
        input_size=input_size,
        hidden_size=5,
        output_size=output_size,
        readout_training='gd',
    )

    inputs = torch.randn(1, batch_size, input_size)
    washout = [0] * batch_size

    forward_output, forward_hidden = model(inputs, washout=washout)

    step_output, step_hidden = model.step(inputs[0])

    assert torch.allclose(step_output, forward_output)
    assert torch.allclose(step_hidden, forward_hidden)


def test_predict_autoregressive_shapes():
    torch.manual_seed(0)
    batch_size = 2
    input_size = 2
    output_size = 2
    steps = 4

    model = ESN(
        input_size=input_size,
        hidden_size=4,
        output_size=output_size,
        readout_training='gd',
    )

    init_u = torch.randn(batch_size, input_size)
    outputs, hidden = model.predict_autoregressive(init_u, steps=steps)

    assert outputs.shape == (steps, batch_size, output_size)
    assert hidden is not None
    assert torch.isfinite(outputs).all()
