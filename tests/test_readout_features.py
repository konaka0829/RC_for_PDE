import torch

from torchesn.nn import ESN


def test_readout_features_linear_and_square_dimensions_and_forward():
    torch.manual_seed(0)

    seq_len = 12
    batch_size = 3
    input_size = 4
    hidden_size = 5
    output_size = 2

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        w_io=True,
        readout_training="cholesky",
        readout_features="linear_and_square",
    )

    expected_in_features = input_size + hidden_size * model.num_layers * 2
    assert model.readout.in_features == expected_in_features

    x = torch.randn(seq_len, batch_size, input_size)
    target = torch.randn(seq_len * batch_size, output_size)
    washout = [0] * batch_size

    output, hidden = model(x, washout, target=target)
    assert output is None
    assert hidden is None

    model.fit()

    output, hidden = model(x, washout)
    assert output.shape == (seq_len, batch_size, output_size)
    assert hidden.shape == (model.num_layers, batch_size, hidden_size)
