from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.nn import ESN
from torchesn.utils import prepare_target


def test_readout_features_linear_and_square():
    torch.manual_seed(0)
    seq_len = 5
    batch_size = 2
    input_size = 4
    hidden_size = 3
    output_size = 2
    washout = [1, 1]
    seq_lengths = [seq_len] * batch_size

    model = ESN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        w_io=True,
        readout_features='linear_and_square',
        readout_training='cholesky',
        lambda_reg=1e-3,
    )

    expected_in_features = input_size + hidden_size * 2
    assert model.readout.in_features == expected_in_features

    inputs = torch.randn(seq_len, batch_size, input_size)
    target = torch.randn(seq_len, batch_size, output_size)
    flat_target = prepare_target(
        target,
        seq_lengths=seq_lengths,
        washout=washout,
        batch_first=False,
    )

    output, hidden = model(inputs, washout=washout, target=flat_target)
    assert output is None
    assert hidden is None

    model.fit()

    output, hidden = model(inputs, washout=washout)
    assert output is not None
    assert output.shape[-1] == output_size
    assert hidden is not None
