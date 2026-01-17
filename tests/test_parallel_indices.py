from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from torchesn.nn.parallel_echo_state_network import build_periodic_indices


def test_build_periodic_indices_wraps():
    Q = 6
    g = 2
    l = 1

    indices = build_periodic_indices(Q, g, l)

    assert len(indices) == g
    assert torch.equal(indices[0], torch.tensor([5, 0, 1, 2, 3]))
    assert torch.equal(indices[1], torch.tensor([2, 3, 4, 5, 0]))
