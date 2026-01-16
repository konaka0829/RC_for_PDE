import torch

from torchesn.nn import ESN
from torchesn.utils import prepare_target


def test_linear_and_square_features_support_training_and_inference():
    torch.manual_seed(0)
    input_size = 3
    hidden_size = 5
    num_layers = 2
    output_size = 2
    batch_size = 2
    seq_len = 6
    washout = [1, 2]

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        readout_training="cholesky",
        w_io=True,
        readout_features="linear_and_square",
    )

    expected_in_features = input_size + hidden_size * num_layers * 2
    assert model.readout.in_features == expected_in_features

    inputs = torch.randn(seq_len, batch_size, input_size)
    targets = torch.randn(seq_len, batch_size, output_size)
    seq_lengths = [seq_len] * batch_size
    flat_target = prepare_target(targets, seq_lengths, washout)

    output, hidden = model(inputs, washout, target=flat_target)
    assert output is None
    assert hidden is None

    model.fit()

    output, hidden = model(inputs, washout)
    assert output is not None
    assert hidden is not None
    assert output.shape[-1] == output_size
