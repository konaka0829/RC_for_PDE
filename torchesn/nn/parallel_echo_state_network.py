import torch
import torch.nn as nn

from .echo_state_network import ESN
from ..utils import prepare_target


def build_periodic_indices(Q, g, l):
    """Build periodic wrap indices for parallel ESN inputs.

    Args:
        Q (int): Total number of spatial points.
        g (int): Number of ESN partitions.
        l (int): Overlap size on each side.

    Returns:
        tuple[list[torch.LongTensor], list[torch.LongTensor]]: Input indices and
            center indices for each ESN.
    """
    if Q % g != 0:
        raise ValueError("Q must be divisible by g")
    if l < 0:
        raise ValueError("l must be non-negative")

    q = Q // g
    input_indices = []
    center_indices = []
    for i in range(g):
        start = i * q
        center = torch.arange(start, start + q, dtype=torch.long)
        window = torch.arange(start - l, start + q + l, dtype=torch.long)
        window = window % Q
        input_indices.append(window)
        center_indices.append(center)
    return input_indices, center_indices


class ParallelESN(nn.Module):
    """Parallel ESN ensemble with periodic boundary wrapping.

    Args:
        translation_invariant (bool, optional): If True, share ESN weights across
            all partitions (mu=0 condition). Default: False
        mu (float, optional): Optional parameter to validate translation-invariant
            configuration. If provided and translation_invariant is True, this
            class will enforce mu == 0. When mu is None, the caller is responsible
            for ensuring the mu=0 condition before enabling translation_invariant.
    """

    def __init__(
        self,
        Q,
        g,
        l,
        hidden_size,
        spectral_radius=0.9,
        leaking_rate=1,
        density=1,
        lambda_reg=0,
        readout_training="svd",
        readout_features="linear",
        w_io=False,
        translation_invariant=False,
        mu=None,
        seed=None,
    ):
        super().__init__()
        if Q % g != 0:
            raise ValueError("Q must be divisible by g")
        if translation_invariant and mu is not None and mu != 0:
            raise ValueError("translation_invariant requires mu == 0")
        if seed is not None:
            torch.manual_seed(seed)

        self.Q = Q
        self.g = g
        self.l = l
        self.q = Q // g
        self.translation_invariant = translation_invariant
        self.mu = mu

        input_indices, center_indices = build_periodic_indices(Q, g, l)
        self.input_indices = input_indices
        self.center_indices = center_indices
        self.shared_esn = None
        self.esns = None

        if translation_invariant:
            self.shared_esn = ESN(
                input_size=self.q + 2 * l,
                hidden_size=hidden_size,
                output_size=self.q,
                spectral_radius=spectral_radius,
                leaking_rate=leaking_rate,
                density=density,
                lambda_reg=lambda_reg,
                readout_training=readout_training,
                readout_features=readout_features,
                w_io=w_io,
            )
        else:
            self.esns = nn.ModuleList(
                [
                    ESN(
                        input_size=self.q + 2 * l,
                        hidden_size=hidden_size,
                        output_size=self.q,
                        spectral_radius=spectral_radius,
                        leaking_rate=leaking_rate,
                        density=density,
                        lambda_reg=lambda_reg,
                        readout_training=readout_training,
                        readout_features=readout_features,
                        w_io=w_io,
                    )
                    for _ in range(g)
                ]
            )

    def _extract_local(self, u, indices):
        if u.dim() == 1:
            return u.index_select(0, indices)
        if u.dim() == 2:
            return u.index_select(1, indices)
        raise ValueError("u must be 1D or 2D (time, Q)")

    def fit(self, u_train, washout):
        if u_train.dim() != 2:
            raise ValueError("u_train must have shape (seq_len, Q)")
        if u_train.size(1) != self.Q:
            raise ValueError("u_train.size(1) must be equal to Q")

        if isinstance(washout, int):
            washout_list = [washout] * (self.g if self.translation_invariant else 1)
        else:
            washout_list = list(washout)

        inputs = u_train[:-1]
        targets = u_train[1:]
        seq_len = inputs.size(0)
        seq_lengths = [seq_len]

        if self.translation_invariant:
            local_inputs = [
                self._extract_local(inputs, idx) for idx in self.input_indices
            ]
            local_targets = [
                self._extract_local(targets, idx) for idx in self.center_indices
            ]
            local_inputs = torch.stack(local_inputs, dim=1)
            local_targets = torch.stack(local_targets, dim=1)
            flat_target = prepare_target(
                local_targets, [seq_len] * self.g, washout_list
            )
            self.shared_esn(local_inputs, washout_list, target=flat_target)
            if self.shared_esn.readout_training in {"cholesky", "inv"}:
                self.shared_esn.fit()
            return

        for esn, input_idx, center_idx in zip(
            self.esns, self.input_indices, self.center_indices
        ):
            local_inputs = self._extract_local(inputs, input_idx)
            local_targets = self._extract_local(targets, center_idx)

            local_inputs = local_inputs.unsqueeze(1)
            local_targets = local_targets.unsqueeze(1)
            flat_target = prepare_target(
                local_targets, seq_lengths, washout_list
            )

            esn(local_inputs, washout_list, target=flat_target)
            if esn.readout_training in {"cholesky", "inv"}:
                esn.fit()

    def warmup(self, u_hist, epsilon=0.0):
        if u_hist.dim() != 2:
            raise ValueError("u_hist must have shape (seq_len, Q)")
        if u_hist.size(1) != self.Q:
            raise ValueError("u_hist.size(1) must be equal to Q")

        if self.translation_invariant:
            local_inputs = [
                self._extract_local(u_hist, idx) for idx in self.input_indices
            ]
            local_inputs = torch.stack(local_inputs, dim=1)
            if epsilon != 0:
                local_inputs = local_inputs + epsilon * torch.randn_like(local_inputs)
            _, hidden = self.shared_esn(local_inputs, washout=[0] * self.g)
            return [hidden[:, i : i + 1, :] for i in range(self.g)]

        hx_list = []
        for esn, input_idx in zip(self.esns, self.input_indices):
            local_inputs = self._extract_local(u_hist, input_idx)
            if epsilon != 0:
                local_inputs = local_inputs + epsilon * torch.randn_like(local_inputs)
            local_inputs = local_inputs.unsqueeze(1)
            _, hidden = esn(local_inputs, washout=[0])
            hx_list.append(hidden)

        return hx_list

    def predict(self, u_hist, steps, epsilon=0.0):
        if u_hist.dim() != 2:
            raise ValueError("u_hist must have shape (seq_len, Q)")
        if u_hist.size(1) != self.Q:
            raise ValueError("u_hist.size(1) must be equal to Q")

        hx_list = self.warmup(u_hist, epsilon=epsilon)
        current = u_hist[-1]
        outputs = []

        for _ in range(steps):
            if self.translation_invariant:
                local_inputs = [
                    current.index_select(0, idx) for idx in self.input_indices
                ]
                local_inputs = torch.stack(local_inputs, dim=0)
                if epsilon != 0:
                    local_inputs = local_inputs + epsilon * torch.randn_like(local_inputs)
                output_t, hx_next = self.shared_esn.step(local_inputs, hx=torch.cat(hx_list, dim=1))
                next_chunks = [output_t[i] for i in range(self.g)]
                hx_list = [hx_next[:, i : i + 1, :] for i in range(self.g)]
            else:
                next_chunks = []
                next_hx = []
                for esn, input_idx, hx in zip(self.esns, self.input_indices, hx_list):
                    local_input = current.index_select(0, input_idx)
                    if epsilon != 0:
                        local_input = local_input + epsilon * torch.randn_like(local_input)
                    local_input = local_input.unsqueeze(0)
                    output_t, hx_next = esn.step(local_input, hx=hx)
                    next_chunks.append(output_t.squeeze(0))
                    next_hx.append(hx_next)

                hx_list = next_hx

            current = torch.cat(next_chunks, dim=0)
            outputs.append(current)

        return torch.stack(outputs, dim=0)
