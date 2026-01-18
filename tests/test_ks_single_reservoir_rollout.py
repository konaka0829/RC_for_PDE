import unittest

import torch

from examples.ks_single_reservoir import autoregressive_rollout


class DummyESN:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, inputs, washout, h_0=None):
        self.calls += 1
        output = inputs + 1.0
        hidden = inputs.mean(dim=0, keepdim=True)
        return output, hidden


class AutoregressiveRolloutTests(unittest.TestCase):
    def test_rollout_uses_last_warmup_output(self):
        model = DummyESN()
        warmup_u = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
        prediction = autoregressive_rollout(model, warmup_u, pred_steps=3)

        expected = torch.tensor([[3.0], [4.0], [5.0]])
        self.assertTrue(torch.allclose(prediction, expected))
        self.assertEqual(model.calls, 3)


if __name__ == "__main__":
    unittest.main()
