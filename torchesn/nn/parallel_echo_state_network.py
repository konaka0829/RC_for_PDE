import torch
import torch.nn as nn

from .echo_state_network import ESN
from ..utils import prepare_target


def build_periodic_indices(Q, g, l, device=None):
    if Q % g != 0:
        raise ValueError("Q must be divisible by g.")
    if l < 0:
        raise ValueError("l must be non-negative.")
    q = Q // g
    indices = []
    for i in range(g):
        start = i * q
        window = [(start + offset) % Q for offset in range(-l, q + l)]
        indices.append(torch.tensor(window, device=device, dtype=torch.long))
    return indices


class ParallelESN(nn.Module):
    def __init__(
        self,
        Q,
        g,
        l,
        hidden_size,
        mu=0.0,
        spectral_radius=0.9,
        leaking_rate=1.0,
        density=1.0,
        lambda_reg=0.0,
        readout_training='svd',
        readout_features='linear',
        w_io=False,
        translation_invariant=True,
        seed=None,
    ):
        """Parallel ESN for periodic domains.

        Args:
            Q (int): Total state dimension.
            g (int): Number of partitions.
            l (int): Overlap radius on each side.
            mu (float): Parameter used to decide weight sharing. When mu == 0 and
                translation_invariant is True, all partitions share weights.
            translation_invariant (bool): Enable translation-invariant behavior.
                Weight sharing is applied only when mu == 0.
        """
        super().__init__()
        if Q <= 0 or g <= 0:
            raise ValueError("Q and g must be positive.")
        if Q % g != 0:
            raise ValueError("Q must be divisible by g.")

        self.Q = Q
        self.g = g
        self.l = l
        self.q = Q // g
        self.mu = mu
        self.hidden_size = hidden_size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.density = density
        self.lambda_reg = lambda_reg
        self.readout_training = readout_training
        self.readout_features = readout_features
        self.w_io = w_io
        self.translation_invariant = translation_invariant

        if seed is not None:
            torch.manual_seed(seed)

        input_size = self.q + 2 * self.l
        self.esns = nn.ModuleList()
        self.shared_esn = None
        share_weights = self.translation_invariant and self.mu == 0
        if share_weights:
            if seed is not None:
                torch.manual_seed(seed)
            self.shared_esn = ESN(
                input_size=input_size,
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
            self.esns.append(self.shared_esn)
        else:
            for idx in range(g):
                if seed is not None:
                    torch.manual_seed(seed + idx)
                self.esns.append(
                    ESN(
                        input_size=input_size,
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
                )

    def _get_esn(self, idx):
        if self.shared_esn is not None:
            return self.shared_esn
        return self.esns[idx]

    def _ensure_sequence(self, u):
        if u.dim() == 2:
            u = u.unsqueeze(1)
        if u.dim() != 3:
            raise RuntimeError(
                'u must have shape (seq_len, Q) or (seq_len, batch, Q).')
        if u.size(-1) != self.Q:
            raise RuntimeError(
                'Last dimension of u must be Q={}, got {}.'.format(
                    self.Q, u.size(-1)))
        return u

    def _normalize_washout(self, washout, batch_size):
        if isinstance(washout, int):
            return [washout] * batch_size
        if len(washout) != batch_size:
            raise RuntimeError(
                'washout must have length {}, got {}'.format(
                    batch_size, len(washout)))
        return list(washout)

    def fit(self, u_train, washout, epsilon=0.0):
        u_train = self._ensure_sequence(u_train)
        if u_train.size(0) < 2:
            raise ValueError('u_train must have at least 2 timesteps.')
        seq_len, batch_size, _ = u_train.shape
        washout = self._normalize_washout(washout, batch_size)

        inputs = u_train[:-1]
        targets = u_train[1:]
        if epsilon:
            inputs = inputs + epsilon * torch.randn_like(inputs)

        indices = build_periodic_indices(self.Q, self.g, self.l, device=u_train.device)
        seq_lengths = [inputs.size(0)] * batch_size

        for idx in range(self.g):
            esn = self._get_esn(idx)
            esn_input = inputs.index_select(2, indices[idx])
            target_chunk = targets[:, :, idx * self.q:(idx + 1) * self.q]
            flat_target = prepare_target(
                target_chunk,
                seq_lengths=seq_lengths,
                washout=washout,
                batch_first=False,
            )
            esn(esn_input, washout, target=flat_target)

        for esn in self.esns:
            esn.fit()
        return self

    def warmup(self, u_hist, epsilon=0.0):
        u_hist = self._ensure_sequence(u_hist)
        if u_hist.size(0) < 1:
            raise ValueError('u_hist must have at least 1 timestep.')
        if epsilon:
            u_hist = u_hist + epsilon * torch.randn_like(u_hist)

        indices = build_periodic_indices(self.Q, self.g, self.l, device=u_hist.device)
        hxs = []
        washout = [0] * u_hist.size(1)
        for idx in range(self.g):
            esn = self._get_esn(idx)
            esn_input = u_hist.index_select(2, indices[idx])
            _, hx = esn(esn_input, washout)
            hxs.append(hx)
        return hxs

    def predict(self, u_hist, steps, epsilon=0.0):
        if steps <= 0:
            raise ValueError('steps must be a positive integer.')
        u_hist = self._ensure_sequence(u_hist)
        hxs = self.warmup(u_hist, epsilon=epsilon)
        current_u = u_hist[-1]
        if epsilon:
            current_u = current_u + epsilon * torch.randn_like(current_u)

        outputs = []
        indices = build_periodic_indices(self.Q, self.g, self.l, device=u_hist.device)
        for _ in range(steps):
            step_output = torch.zeros(
                current_u.size(0), self.Q, device=current_u.device, dtype=current_u.dtype)
            for idx in range(self.g):
                esn = self._get_esn(idx)
                esn_input = current_u.index_select(1, indices[idx])
                esn_output, hx_next = esn.step(esn_input, hx=hxs[idx])
                hxs[idx] = hx_next
                esn_output = esn_output.squeeze(0)
                step_output[:, idx * self.q:(idx + 1) * self.q] = esn_output
            outputs.append(step_output)
            current_u = step_output
            if epsilon:
                current_u = current_u + epsilon * torch.randn_like(current_u)

        outputs = torch.stack(outputs, dim=0)
        return outputs
