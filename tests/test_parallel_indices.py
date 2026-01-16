import torch

from torchesn.nn.parallel_echo_state_network import build_periodic_indices


def test_build_periodic_indices_wraps_boundaries():
    input_indices, center_indices = build_periodic_indices(Q=8, g=2, l=1)

    assert len(input_indices) == 2
    assert len(center_indices) == 2

    expected_first = torch.tensor([7, 0, 1, 2, 3, 4])
    expected_second = torch.tensor([3, 4, 5, 6, 7, 0])

    torch.testing.assert_close(input_indices[0], expected_first)
    torch.testing.assert_close(input_indices[1], expected_second)

    torch.testing.assert_close(center_indices[0], torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(center_indices[1], torch.tensor([4, 5, 6, 7]))
