import torch

from torchesn.nn import ESN
from torchesn.utils import prepare_target


def test_prepare_target_preserves_dtype():
    target = torch.randn(4, 2, 3, dtype=torch.float64)
    seq_lengths = [4, 4]
    washout = [0, 0]
    prepared = prepare_target(target, seq_lengths, washout)
    assert prepared.dtype == torch.float64


def test_svd_path_preserves_dtype():
    torch.manual_seed(0)
    seq_len = 3
    batch_size = 1
    input_size = 2
    output_size = 2

    model = ESN(
        input_size=input_size,
        hidden_size=4,
        output_size=output_size,
        readout_training="svd",
        readout_features="linear",
    ).to(dtype=torch.float64)

    x = torch.randn(seq_len, batch_size, input_size, dtype=torch.float64)
    y = torch.randn(seq_len, batch_size, output_size, dtype=torch.float64)
    target = prepare_target(y, [seq_len], [0])

    model(x, [0], target=target)

    assert model.readout.weight.dtype == torch.float64
