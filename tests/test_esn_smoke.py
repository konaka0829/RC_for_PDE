import torch

from torchesn.nn import ESN
from torchesn.utils import prepare_target


def test_esn_cholesky_offline_fit_forward():
    torch.manual_seed(0)
    seq_len = 6
    batch_size = 2
    input_size = 3
    hidden_size = 5
    output_size = 2
    washout = [1, 2]

    inputs = torch.randn(seq_len, batch_size, input_size)
    targets = torch.randn(seq_len, batch_size, output_size)

    seq_lengths = [seq_len] * batch_size
    flat_target = prepare_target(targets, seq_lengths, washout)

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        readout_training="cholesky",
    )

    output, hidden = model(inputs, washout, target=flat_target)
    assert output is None
    assert hidden is None

    model.fit()

    output, hidden = model(inputs, washout)
    assert output is not None
    assert hidden is not None
