import unittest

import torch
import torch.nn as nn

from torchesn.nn.echo_state_network import ESN


class DummyReservoir(nn.Module):
    def __init__(self, output, hidden):
        super().__init__()
        self._output = output
        self._hidden = hidden

    def forward(self, input, h_0=None):
        return self._output, self._hidden


class FeatureTransformTests(unittest.TestCase):
    def test_square_even_transform(self):
        model = ESN(1, 4, 1, feature_transform='square_even')
        features = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                                 [4.0, 5.0, 6.0, 7.0]])
        transformed = model._apply_feature_transform(features)
        expected = features.clone()
        expected[..., 1::2] = expected[..., 1::2] ** 2
        self.assertTrue(torch.equal(transformed, expected))

    def test_offline_design_matrix_bias_untouched(self):
        output = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ])
        hidden = torch.zeros(1, 1, 4)
        model = ESN(
            1,
            4,
            1,
            readout_training='cholesky',
            output_steps='all',
            feature_transform='square_even',
        )
        model.reservoir = DummyReservoir(output, hidden)
        input_tensor = torch.zeros(2, 1, 1)
        target = torch.zeros(2, 1)

        model(input_tensor, washout=[0], target=target)

        expected = output.clone()
        expected[..., 1::2] = expected[..., 1::2] ** 2
        self.assertTrue(torch.equal(model.X[:, 0], torch.ones(2)))
        self.assertTrue(torch.equal(model.X[:, 1:], expected[:, 0, :]))

    def test_offline_design_matrix_no_bias_column(self):
        output = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ])
        hidden = torch.zeros(1, 1, 4)
        model = ESN(
            1,
            4,
            1,
            readout_training='cholesky',
            output_steps='all',
            feature_transform='square_even',
            readout_bias=False,
        )
        model.reservoir = DummyReservoir(output, hidden)
        input_tensor = torch.zeros(2, 1, 1)
        target = torch.zeros(2, 1)

        model(input_tensor, washout=[0], target=target)

        expected = output.clone()
        expected[..., 1::2] = expected[..., 1::2] ** 2
        self.assertEqual(tuple(model.X.shape), (2, 4))
        self.assertTrue(torch.equal(model.X, expected[:, 0, :]))


if __name__ == '__main__':
    unittest.main()
