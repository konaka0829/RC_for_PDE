import unittest

import torch

from torchesn.nn.reservoir import Reservoir


class BlockWinTests(unittest.TestCase):
    def test_block_win_structure(self):
        torch.manual_seed(1234)
        input_size = 4
        hidden_size = 12
        sigma = 0.5
        reservoir = Reservoir(
            'RES_TANH',
            input_size,
            hidden_size,
            num_layers=1,
            leaking_rate=1.0,
            spectral_radius=0.9,
            w_ih_scale=torch.ones(input_size + 1),
            density=1.0,
            input_init='block',
            win_sigma=sigma,
        )
        weight = reservoir.weight_ih_l0
        block_size = hidden_size // input_size
        expected_mask = torch.zeros_like(weight, dtype=torch.bool)
        for i in range(input_size):
            rows = slice(i * block_size, (i + 1) * block_size)
            expected_mask[rows, i] = True

        self.assertTrue(torch.all(weight[~expected_mask] == 0))
        self.assertTrue(torch.all(weight[expected_mask] <= sigma))
        self.assertTrue(torch.all(weight[expected_mask] >= -sigma))

    def test_block_win_invalid_hidden_size(self):
        with self.assertRaises(ValueError):
            Reservoir(
                'RES_TANH',
                4,
                10,
                num_layers=1,
                leaking_rate=1.0,
                spectral_radius=0.9,
                w_ih_scale=torch.ones(5),
                density=1.0,
                input_init='block',
            )


if __name__ == '__main__':
    unittest.main()
