import unittest
import torch

from examples.ks_single_reservoir import autoregressive_rollout


class DummyModel:
    """
    ESN 互換の __call__ を持つダミーモデル。
    返す output は常に input + 1。
    呼び出し時の input を記録して、rollout が何を入力しているか検査する。
    """
    def __init__(self):
        self.calls = []

    def __call__(self, input, washout, h_0=None, target=None):
        self.calls.append(input.clone())
        out = input + 1.0
        hidden = torch.zeros(1)
        return out, hidden


class TestKSRolloutAlignment(unittest.TestCase):
    def test_rollout_uses_last_warmup_output_as_first_prediction(self):
        model = DummyModel()
        warmup_u = torch.tensor(
            [
                [[10.0, 20.0]],
                [[30.0, 40.0]],
            ]
        )  # shape (2, 1, 2)

        pred = autoregressive_rollout(model, warmup_u, pred_steps=2)
        self.assertEqual(tuple(pred.shape), (2, 2))

        # First prediction must be last warmup output = last warmup input + 1
        self.assertTrue(torch.equal(pred[0], warmup_u[-1, 0] + 1.0))
        # Second prediction must be +2
        self.assertTrue(torch.equal(pred[1], warmup_u[-1, 0] + 2.0))

        # Call 1: warmup sequence
        self.assertTrue(torch.equal(model.calls[0], warmup_u))
        # Call 2: first closed-loop input must be warmup_out[-1:] = warmup_u[-1:] + 1
        self.assertTrue(torch.equal(model.calls[1], warmup_u[-1:, :, :] + 1.0))

    def test_pred_steps_one(self):
        model = DummyModel()
        warmup_u = torch.tensor([[[1.0, 2.0]]])  # (1,1,2)
        pred = autoregressive_rollout(model, warmup_u, pred_steps=1)
        self.assertEqual(tuple(pred.shape), (1, 2))
        self.assertTrue(torch.equal(pred[0], torch.tensor([2.0, 3.0])))


if __name__ == "__main__":
    unittest.main()
