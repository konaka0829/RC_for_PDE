from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from .reservoir_utils import generate_reservoir_sparse, make_block_input_weights
from .solver import _as_device


@torch.no_grad()
def _augment_even_square_inplace(X: torch.Tensor) -> torch.Tensor:
    """Apply MATLAB's even-index squaring feature map.

    Python 0-based indices: 1,3,5,... are squared.
    Works for vectors [N] and matrices [N, T].
    """
    X[1::2].square_()
    return X


@dataclass(frozen=True)
class KSParallelParams:
    """Parameter bundle matching the MATLAB `resparams` structure."""

    # Global spatiotemporal system dimension
    Q: int

    # Parallelization
    g: int
    locality: int

    # Reservoir
    approx_reservoir_size: int = 5000
    radius: float = 0.6
    degree: float = 3.0
    beta: float = 1e-4

    # Time lengths
    discard_length: int = 1000
    train_length: int = 79_000
    predict_length: int = 2999

    # Input scaling (MATLAB script multiplies the data by sigma)
    sigma: float = 0.5

    # Randomization
    jobid: int = 1


def _build_local_indices(Q: int, g: int, locality: int) -> Tuple[int, int, List[List[int]]]:
    """Return (chunk_size, Din, indices_per_reservoir).

    Reservoir i (0-based) is responsible for global indices:
      chunk = [i*chunk_size, ..., (i+1)*chunk_size-1]
    and sees inputs:
      [rear_overlap (l), chunk (q), forward_overlap (l)]
    with periodic wrap.
    """
    if Q <= 0:
        raise ValueError("Q must be positive")
    if g <= 0:
        raise ValueError("g must be positive")
    if Q % g != 0:
        raise ValueError("Q must be divisible by g")
    if locality < 0:
        raise ValueError("locality must be >= 0")

    chunk_size = Q // g
    if locality > chunk_size:
        raise ValueError("locality must be <= chunk_size (overlap cannot exceed chunk)")

    Din = chunk_size + 2 * locality
    indices: List[List[int]] = []
    for i in range(g):
        begin = i * chunk_size
        end = begin + chunk_size - 1
        rear = [((begin - locality) + k) % Q for k in range(locality)]
        chunk = list(range(begin, end + 1))
        front = [((end + 1) + k) % Q for k in range(locality)]
        indices.append(rear + chunk + front)

    return chunk_size, Din, indices


class KSParallelReservoir(torch.nn.Module):
    """Parallel reservoir system reproducing MATLAB `KSParallelReservoir`.

    Typical workflow:
        model = KSParallelReservoir(params)
        model.fit(u_train)  # u_train: [Q, T_train]
        pred = model.predict_intervals(u_test, warmup_starts=[...], sync_length=32)

    Shapes:
        - Global data u: [Q, T]
        - Per-reservoir input window h_i: [Din, T]
        - Per-reservoir output g_i: [chunk_size, T]
    """

    def __init__(
        self,
        params: KSParallelParams,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
        power_iter: int = 100,
    ) -> None:
        super().__init__()

        dev = _as_device(device)

        chunk_size, Din, idx = _build_local_indices(params.Q, params.g, params.locality)

        nodes_per_input = int(round(float(params.approx_reservoir_size) / float(Din)))
        nodes_per_input = max(nodes_per_input, 1)

        N = nodes_per_input * Din

        self.params = params
        self.device = dev
        self.dtype = dtype
        self.power_iter = int(power_iter)

        self.Q = int(params.Q)
        self.g = int(params.g)
        self.locality = int(params.locality)
        self.chunk_size = int(chunk_size)
        self.Din = int(Din)
        self.nodes_per_input = int(nodes_per_input)
        self.N = int(N)

        w_in_diag = make_block_input_weights(
            reservoir_size=self.N,
            num_inputs=self.Din,
            sigma=1.0,
            device=self.device,
            dtype=self.dtype,
        )
        self.register_buffer("w_in_diag", w_in_diag)

        self.As: List[torch.Tensor] = []
        self.w_outs: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []

        for i in range(self.g):
            seed_A = int((i + 1) + params.jobid)
            A_i = generate_reservoir_sparse(
                size=self.N,
                radius=float(params.radius),
                degree=float(params.degree),
                seed=seed_A,
                device=self.device,
                dtype=self.dtype,
                power_iter=self.power_iter,
            )
            self.As.append(A_i)
            self.w_outs.append(torch.zeros((self.chunk_size, self.N), device=self.device, dtype=self.dtype))
            self.states.append(torch.zeros((self.N,), device=self.device, dtype=self.dtype))

        self._indices_per_reservoir = idx
        self._central_slice = slice(self.locality, self.locality + self.chunk_size)

    @torch.no_grad()
    def _input_term(self, u: torch.Tensor) -> torch.Tensor:
        """Compute Win @ u efficiently for the block-structured Win."""
        if u.ndim != 1 or u.shape[0] != self.Din:
            raise ValueError(f"u must have shape [{self.Din}]")
        u_rep = u.repeat_interleave(self.nodes_per_input)
        return self.w_in_diag * u_rep

    @torch.no_grad()
    def _step(self, i: int, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """x <- tanh(A_i x + Win u)."""
        Ax = torch.sparse.mm(self.As[i], x.unsqueeze(1)).squeeze(1)
        return torch.tanh(Ax + self._input_term(u))

    def _ensure_global_shape(self, u: torch.Tensor) -> torch.Tensor:
        """Accept [Q,T] or [T,Q] and return [Q,T] on device/dtype."""
        if u.ndim != 2:
            raise ValueError("u must be 2D")
        if u.shape[0] == self.Q:
            U = u
        elif u.shape[1] == self.Q:
            U = u.T
        else:
            raise ValueError(f"u must have shape [Q,T] or [T,Q] with Q={self.Q}")
        return U.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def fit(self, u_train: torch.Tensor, *, identical_reservoirs: bool = False, time_chunk: int = 256) -> "KSParallelReservoir":
        """Train all reservoirs' readouts via ridge regression."""
        U = self._ensure_global_shape(u_train)
        T = int(U.shape[1])
        need = int(self.params.discard_length + self.params.train_length)
        if T < need:
            raise ValueError(f"Need at least discard_length+train_length={need} time steps, got {T}")
        if time_chunk <= 0:
            raise ValueError("time_chunk must be positive")

        def train_one(res_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            loc_idx = self._indices_per_reservoir[res_idx]
            data = (float(self.params.sigma) * U[loc_idx, :need]).contiguous()

            x = torch.zeros((self.N,), device=self.device, dtype=self.dtype)

            for t in range(int(self.params.discard_length)):
                x = self._step(res_idx, x, data[:, t])

            S = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)
            YX = torch.zeros((self.chunk_size, self.N), device=self.device, dtype=self.dtype)

            train_len = int(self.params.train_length)
            disc = int(self.params.discard_length)
            processed = 0

            while processed < train_len:
                L = min(int(time_chunk), train_len - processed)

                X = torch.empty((self.N, L), device=self.device, dtype=self.dtype)

                for j in range(L):
                    X[:, j] = x
                    global_j = processed + j
                    if global_j < train_len - 1:
                        x = self._step(res_idx, x, data[:, disc + global_j])

                Phi = X
                _augment_even_square_inplace(Phi)

                Y = data[self._central_slice, disc + processed : disc + processed + L]

                S.addmm_(Phi, Phi.T)
                YX.addmm_(Y, Phi.T)

                processed += L

            beta = float(self.params.beta)
            if beta > 0:
                S = S + beta * torch.eye(self.N, device=self.device, dtype=self.dtype)

            Lchol = torch.linalg.cholesky(S)
            w_out_T = torch.cholesky_solve(YX.T, Lchol)
            w_out = w_out_T.T.contiguous()

            return w_out, x

        if identical_reservoirs:
            w_out0, _ = train_one(0)
            A0 = self.As[0]
            for i in range(self.g):
                self.As[i] = A0
                self.w_outs[i] = w_out0
            return self

        for i in range(self.g):
            w_out, _ = train_one(i)
            self.w_outs[i] = w_out

        return self

    @torch.no_grad()
    def predict_one_interval(
        self,
        u_test: torch.Tensor,
        *,
        warmup_start: int,
        sync_length: int = 32,
        predict_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict a single interval using synchronous neighbour exchange."""
        U = self._ensure_global_shape(u_test)
        T = int(U.shape[1])

        pl = int(self.params.predict_length if predict_length is None else predict_length)
        if pl <= 0:
            raise ValueError("predict_length must be positive")
        if sync_length < 0:
            raise ValueError("sync_length must be >= 0")

        if warmup_start < 0 or warmup_start + sync_length + pl > T:
            raise ValueError("warmup_start+sync_length+predict_length must fit within u_test")

        states = [torch.zeros((self.N,), device=self.device, dtype=self.dtype) for _ in range(self.g)]

        sigma = float(self.params.sigma)
        for t in range(int(sync_length)):
            tt = int(warmup_start + t)
            for i in range(self.g):
                loc_idx = self._indices_per_reservoir[i]
                u_loc = sigma * U[loc_idx, tt]
                states[i] = self._step(i, states[i], u_loc)

        pred = torch.empty((self.Q, pl), device=self.device, dtype=self.dtype)

        for k in range(pl):
            outs: List[torch.Tensor] = []
            for i in range(self.g):
                x_aug = states[i].clone()
                _augment_even_square_inplace(x_aug)
                outs.append(self.w_outs[i] @ x_aug)

            pred[:, k] = torch.cat(outs, dim=0)

            new_states: List[torch.Tensor] = []
            for i in range(self.g):
                if self.locality > 0:
                    rear = outs[(i - 1) % self.g][-self.locality :]
                    front = outs[(i + 1) % self.g][: self.locality]
                    feedback = torch.cat([rear, outs[i], front], dim=0)
                else:
                    feedback = outs[i]
                new_states.append(self._step(i, states[i], feedback))

            states = new_states

        return pred

    @torch.no_grad()
    def predict_intervals(
        self,
        u_test: torch.Tensor,
        *,
        warmup_starts: Sequence[int],
        sync_length: int = 32,
        predict_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Predict multiple non-overlapping intervals and concatenate outputs."""
        pl = int(self.params.predict_length if predict_length is None else predict_length)
        preds = [
            self.predict_one_interval(
                u_test,
                warmup_start=int(s),
                sync_length=sync_length,
                predict_length=pl,
            )
            for s in warmup_starts
        ]
        return torch.cat(preds, dim=1)
