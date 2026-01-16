import torch

from torchesn.nn import ESN


def test_esn_smoke_cholesky_forward_fit_forward():
    torch.manual_seed(0)

    seq_len = 30
    batch_size = 4
    input_size = 3
    hidden_size = 8
    output_size = 2

    x = torch.randn(seq_len, batch_size, input_size)
    target = torch.randn(seq_len * batch_size, output_size)

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        readout_training="cholesky",
    )

    washout = [0] * batch_size

    output, hidden = model(x, washout, target=target)
    assert output is None
    assert hidden is None

    output, hidden = model(x, washout, target=target)
    assert output is None
    assert hidden is None

    model.fit()

    output, hidden = model(x, washout)

    assert output.shape == (seq_len, batch_size, output_size)
    assert hidden.shape == (model.num_layers, batch_size, hidden_size)
