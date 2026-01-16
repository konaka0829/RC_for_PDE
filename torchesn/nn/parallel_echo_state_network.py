import torch
import torch.nn as nn

from .echo_state_network import ESN
from ..utils import prepare_target


def parallel_input_indices(Q, g, l, i):
    """Compute periodic input indices for the i-th reservoir."""
    if Q % g != 0:
        raise ValueError("Q must be divisible by g.")
    q = Q // g
    start = i * q - l
    length = q + 2 * l
    return [(start + offset) % Q for offset in range(length)]


def parallel_output_indices(Q, g, i):
    """Compute output indices for the i-th reservoir."""
    if Q % g != 0:
        raise ValueError("Q must be divisible by g.")
    q = Q // g
    start = i * q
    return list(range(start, start + q))


class ParallelESN(nn.Module):
    """Parallel reservoirs for 1D periodic grids using ESNs."""

    def __init__(
        self,
        Q,
        g,
        l,
        hidden_size,
        spectral_radius=0.9,
        leaking_rate=1.0,
        density=1.0,
        lambda_reg=0.0,
        readout_training="cholesky",
        readout_features="linear_and_square",
        w_io=False,
        seed=None,
        translation_invariant=False,
        mu=0.0,
    ):
        super().__init__()
        if Q % g != 0:
            raise ValueError("Q must be divisible by g.")
        if g <= 0:
            raise ValueError("g must be positive.")
        if l < 0:
            raise ValueError("l must be non-negative.")
        if translation_invariant and mu != 0:
            raise ValueError("translation_invariant requires mu == 0.")

        if seed is not None:
            torch.manual_seed(seed)

        self.Q = Q
        self.g = g
        self.l = l
        self.q = Q // g
        self.translation_invariant = translation_invariant

        input_size = self.q + 2 * l
        output_size = self.q

        if translation_invariant:
            esn = ESN(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                spectral_radius=spectral_radius,
                leaking_rate=leaking_rate,
                density=density,
                lambda_reg=lambda_reg,
                readout_training=readout_training,
                readout_features=readout_features,
                w_io=w_io,
                output_steps="all",
            )
            self.esn_shared = esn
            self.esns = nn.ModuleList([esn for _ in range(g)])
        else:
            self.esns = nn.ModuleList([
                ESN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    spectral_radius=spectral_radius,
                    leaking_rate=leaking_rate,
                    density=density,
                    lambda_reg=lambda_reg,
                    readout_training=readout_training,
                    readout_features=readout_features,
                    w_io=w_io,
                    output_steps="all",
                )
                for _ in range(g)
            ])

        self.hx = [None for _ in range(g)]

    def _gather_inputs(self, u_batch, i):
        indices = parallel_input_indices(self.Q, self.g, self.l, i)
        return u_batch[:, indices]

    def fit(self, u_train, washout):
        """Fit all reservoirs with offline readout training."""
        if u_train.dim() != 2 or u_train.size(1) != self.Q:
            raise RuntimeError("u_train must have shape (T, Q).")

        T = u_train.size(0)
        washout_list = [washout]
        seq_lengths = [T]

        for i, esn in enumerate(self.esns):
            x_i = u_train[:, parallel_input_indices(self.Q, self.g, self.l, i)]
            y_i = u_train[:, parallel_output_indices(self.Q, self.g, i)]

            x_i = x_i.unsqueeze(1)
            y_i = y_i.unsqueeze(1)
            target = prepare_target(y_i, seq_lengths, washout_list)

            esn(x_i, washout_list, target=target)

            if not self.translation_invariant:
                esn.fit()

        if self.translation_invariant:
            self.esn_shared.fit()

    def warmup(self, u_hist, epsilon):
        """Warm up reservoir states using ground-truth history."""
        if u_hist.dim() != 2 or u_hist.size(1) != self.Q:
            raise RuntimeError("u_hist must have shape (T, Q).")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative.")

        self.hx = [None for _ in range(self.g)]
        if epsilon == 0:
            return

        history = u_hist[-epsilon:]
        for t in range(history.size(0)):
            u_t = history[t].unsqueeze(0)
            for i, esn in enumerate(self.esns):
                input_t = self._gather_inputs(u_t, i)
                _, self.hx[i] = esn.step(input_t, hx=self.hx[i])

    def predict(self, u_hist, steps, epsilon=10):
        """Predict future steps in closed loop."""
        if u_hist.dim() != 2 or u_hist.size(1) != self.Q:
            raise RuntimeError("u_hist must have shape (T, Q).")
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")

        self.warmup(u_hist, epsilon)

        u_current = u_hist[-1].unsqueeze(0)
        outputs = []
        for _ in range(steps):
            segments = []
            for i, esn in enumerate(self.esns):
                input_t = self._gather_inputs(u_current, i)
                output_i, self.hx[i] = esn.step(input_t, hx=self.hx[i])
                segments.append(output_i)
            u_current = torch.cat(segments, dim=-1)
            outputs.append(u_current.squeeze(0))

        return torch.stack(outputs, dim=0)

    def evaluate_windows(self, u_eval, K, tau, epsilon):
        """Compute per-step RMSE curves over sliding windows."""
        if u_eval.dim() != 2 or u_eval.size(1) != self.Q:
            raise RuntimeError("u_eval must have shape (T, Q).")
        if K <= 0 or tau <= 0:
            raise ValueError("K and tau must be positive integers.")

        rmse_curves = []
        for k in range(K):
            start = k * tau
            history_len = max(epsilon, 1)
            hist_end = start + history_len
            target_start = start + (epsilon if epsilon > 0 else 1)
            target_end = target_start + tau
            if target_end > u_eval.size(0):
                break
            u_hist = u_eval[start:hist_end]
            pred = self.predict(u_hist, steps=tau, epsilon=epsilon)
            target = u_eval[target_start:target_end]
            rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=1))
            rmse_curves.append(rmse)

        if not rmse_curves:
            raise ValueError("Not enough data for the requested windows.")

        return torch.stack(rmse_curves, dim=0).mean(dim=0)
