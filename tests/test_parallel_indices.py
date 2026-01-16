from torchesn.nn.parallel_echo_state_network import (
    parallel_input_indices,
    parallel_output_indices,
)


def test_parallel_indices_wrap():
    Q = 8
    g = 2
    l = 1

    indices_0 = parallel_input_indices(Q, g, l, 0)
    indices_1 = parallel_input_indices(Q, g, l, 1)

    assert len(indices_0) == Q // g + 2 * l
    assert indices_0 == [7, 0, 1, 2, 3, 4]
    assert indices_1 == [3, 4, 5, 6, 7, 0]

    outputs_0 = parallel_output_indices(Q, g, 0)
    outputs_1 = parallel_output_indices(Q, g, 1)
    assert outputs_0 == [0, 1, 2, 3]
    assert outputs_1 == [4, 5, 6, 7]
