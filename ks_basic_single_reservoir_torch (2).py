"""PyTorch re-implementation of the MATLAB `KSBasicSingleReservoir` code.

This module mirrors the original repository structure:

- `generate_reservoir.m`  -> `generate_reservoir_sparse`
- `train_reservoir.m`     -> `KSBasicSingleReservoir.fit`
- `reservoir_layer.m`     -> internal state evolution in `fit`
- `train.m`               -> ridge regression with even-index squaring
- `predict.m`             -> `KSBasicSingleReservoir.predict`
- `kursiv_solve.m`        -> `kursiv_solve_etdrk4`

The key (nonstandard) detail from the MATLAB code that is preserved here:

1) **Input weights are block-structured**:
   the reservoir is partitioned into equal blocks, each driven by exactly one
   input variable.

2) **Readout features use alternating squaring**:
   even MATLAB indices (2,4,6,...) are squared before linear regression.
   In 0-based Python indexing this corresponds to indices 1,3,5,...

These two choices are important to reproduce the behaviour of the provided
MATLAB implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import torch


def _as_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


@torch.no_grad()
def estimate_spectral_radius_power_iteration(
    A: torch.Tensor,
    n_iter: int = 100,
    tol: float = 1e-6,
    device: Optional[torch.device | str] = None,
) -> float:
    """Estimate |lambda_max| for a **real** sparse matrix using power iteration.

    This is a practical replacement for MATLAB's `max(abs(eigs(A)))`.

    Args:
        A: Square sparse COO tensor (shape [N,N]). Must be coalesced.
        n_iter: Maximum iterations.
        tol: Early stopping threshold on relative change.
        device: Device for the iterate vector.

    Returns:
        Estimated spectral radius as a Python float.
    """
    if not A.is_sparse:
        raise TypeError("A must be a sparse tensor")
    if A.layout != torch.sparse_coo:
        raise TypeError("A must be a sparse COO tensor")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    dev = _as_device(device) if device is not None else A.device
    N = A.shape[0]
    v = torch.rand(N, device=dev, dtype=torch.float64)
    v = v / (v.norm() + 1e-12)

    last = None
    for _ in range(int(n_iter)):
        Av = torch.sparse.mm(A.to(dev), v.unsqueeze(1)).squeeze(1)
        norm = Av.norm()
        if norm <= 0:
            return 0.0
        v = Av / norm
        est = float(norm)
        if last is not None:
            if abs(est - last) / (abs(last) + 1e-12) < tol:
                return est
        last = est
    return float(last) if last is not None else 0.0


@torch.no_grad()
def generate_reservoir_sparse(
    size: int,
    radius: float,
    degree: float,
    *,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    power_iter: int = 100,
) -> torch.Tensor:
    """Generate a sparse reservoir adjacency matrix like MATLAB `sprand` + scaling.

    MATLAB code:
        sparsity = degree/size;
        A = sprand(size, size, sparsity);
        e = max(abs(eigs(A)));
        A = (A./e).*radius;

    Notes:
        - `sprand` uses i.i.d. uniform values in (0,1) at the nonzero entries.
        - We sample ~degree*size nonzeros (matching sprand's expectation).

    Args:
        size: Reservoir dimension (N).
        radius: Target spectral radius.
        degree: Expected out-degree (kappa in the PRL text).
        seed: Random seed.
        device: Tensor device.
        dtype: Tensor dtype for values.
        power_iter: Iterations for spectral radius estimation.

    Returns:
        Sparse COO tensor A of shape [size,size] with estimated spectral radius `radius`.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if degree <= 0:
        raise ValueError("degree must be positive")

    dev = _as_device(device)
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(int(seed))

    # sprand expectation: nnz ≈ sparsity * size^2 = (degree/size) * size^2 = degree*size
    nnz = int(round(float(degree) * int(size)))
    rows = torch.randint(0, size, (nnz,), generator=g, device=dev, dtype=torch.int64)
    cols = torch.randint(0, size, (nnz,), generator=g, device=dev, dtype=torch.int64)
    vals = torch.rand(nnz, generator=g, device=dev, dtype=dtype)  # ~U(0,1)

    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0),
        vals,
        (size, size),
        device=dev,
        dtype=dtype,
    ).coalesce()

    # Scale to desired spectral radius.
    e = estimate_spectral_radius_power_iteration(A, n_iter=power_iter, device=dev)
    if e <= 0:
        raise RuntimeError("Failed to estimate spectral radius (got <= 0).")
    scale = float(radius) / float(e)
    A = torch.sparse_coo_tensor(A.indices(), A.values() * scale, A.shape, device=dev, dtype=dtype)
    return A.coalesce()


@torch.no_grad()
def make_block_input_weights(
    reservoir_size: int,
    num_inputs: int,
    sigma: float,
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Reproduce the MATLAB block-structured `win` construction.

    MATLAB (train_reservoir.m):
        q = N/num_inputs;
        win = zeros(N,num_inputs);
        for i=1:num_inputs
            rng(i)
            ip = sigma*(-1 + 2*rand(q,1));
            win((i-1)*q+1:i*q, i) = ip;
        end

    Instead of forming a dense [N, Din] matrix, we return a length-N vector
    `w_in_diag` such that:
        win @ u   ==   w_in_diag * repeat_interleave(u, q)

    Args:
        reservoir_size: N.
        num_inputs: Din.
        sigma: Input scaling.

    Returns:
        w_in_diag: shape [N] tensor.
    """
    if reservoir_size % num_inputs != 0:
        raise ValueError("reservoir_size must be divisible by num_inputs")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    dev = _as_device(device)
    q = reservoir_size // num_inputs
    w = torch.empty(reservoir_size, device=dev, dtype=dtype)
    for i in range(num_inputs):
        gen = torch.Generator(device=dev)
        gen.manual_seed(i + 1)  # MATLAB is 1-indexed and uses rng(i)
        ip = (torch.rand(q, generator=gen, device=dev, dtype=dtype) * 2.0 - 1.0) * float(sigma)
        w[i * q : (i + 1) * q] = ip
    return w


@dataclass
class KSReservoirParams:
    """Parameter bundle matching the MATLAB scripts."""

    # Reservoir
    radius: float = 0.6
    degree: float = 3.0
    sigma: float = 1.0
    beta: float = 1e-4
    reservoir_size: int = 5000

    # Training/prediction lengths
    train_length: int = 70_000
    predict_length: int = 1_000


class KSBasicSingleReservoir(torch.nn.Module):
    """Single-reservoir ESN matching the MATLAB `KSBasicSingleReservoir` implementation."""

    def __init__(
        self,
        *,
        num_inputs: int,
        reservoir_size: int,
        radius: float = 0.6,
        degree: float = 3.0,
        sigma: float = 1.0,
        beta: float = 1e-4,
        seed_A: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float64,
        power_iter: int = 100,
    ) -> None:
        super().__init__()
        if num_inputs <= 0:
            raise ValueError("num_inputs must be positive")
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be positive")
        if reservoir_size % num_inputs != 0:
            raise ValueError("reservoir_size must be divisible by num_inputs")
        if beta < 0:
            raise ValueError("beta must be non-negative")

        self.num_inputs = int(num_inputs)
        self.N = int(reservoir_size)
        self.q = self.N // self.num_inputs
        self.radius = float(radius)
        self.degree = float(degree)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.dtype = dtype
        self.device = _as_device(device)

        A = generate_reservoir_sparse(
            self.N,
            self.radius,
            self.degree,
            seed=seed_A,
            device=self.device,
            dtype=self.dtype,
            power_iter=power_iter,
        )
        w_in_diag = make_block_input_weights(
            self.N,
            self.num_inputs,
            self.sigma,
            device=self.device,
            dtype=self.dtype,
        )

        # Use buffers (not trainable by gradient).
        self.register_buffer("A", A)
        self.register_buffer("w_in_diag", w_in_diag)
        self.register_buffer("w_out", torch.zeros(self.num_inputs, self.N, device=self.device, dtype=self.dtype))
        self.register_buffer("state", torch.zeros(self.N, device=self.device, dtype=self.dtype))

    @torch.no_grad()
    def reset_state(self) -> None:
        self.state.zero_()

    @torch.no_grad()
    def _augment_state_inplace(self, X: torch.Tensor) -> torch.Tensor:
        """Apply MATLAB's even-index squaring feature map.

        MATLAB: states(2:2:N,:) = states(2:2:N,:).^2
        Python: indices 1,3,5,... are squared.
        """
        X[1::2].square_()
        return X

    @torch.no_grad()
    def _input_term(self, u: torch.Tensor) -> torch.Tensor:
        """Compute win*u efficiently for the block-structured Win."""
        if u.ndim != 1 or u.shape[0] != self.num_inputs:
            raise ValueError(f"u must have shape [{self.num_inputs}]")
        u_rep = u.repeat_interleave(self.q)
        return self.w_in_diag * u_rep

    @torch.no_grad()
    def step(self, u: torch.Tensor) -> torch.Tensor:
        """Teacher-forced reservoir update: x <- tanh(Ax + Win*u)."""
        Ax = torch.sparse.mm(self.A, self.state.unsqueeze(1)).squeeze(1)
        self.state = torch.tanh(Ax + self._input_term(u))
        return self.state

    @torch.no_grad()
    def fit(self, u_train: torch.Tensor, *, chunk_size: int = 2048) -> "KSBasicSingleReservoir":
        """Train Wout by ridge regression (closed form), matching `train.m`.

        Args:
            u_train: shape [Din, T]. (Din == num_inputs)
            chunk_size: time chunking to limit temporary memory.

        Returns:
            self
        """
        if u_train.ndim != 2 or u_train.shape[0] != self.num_inputs:
            raise ValueError(f"u_train must have shape [{self.num_inputs}, T]")
        T = int(u_train.shape[1])
        if T < 2:
            raise ValueError("Need at least 2 time steps for training")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        u_train = u_train.to(device=self.device, dtype=self.dtype)
        self.reset_state()

        # Accumulate S = X X^T and YX = Y X^T where X are augmented states.
        S = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)
        YX = torch.zeros((self.num_inputs, self.N), device=self.device, dtype=self.dtype)

        t = 0
        x = self.state

        while t < T:
            end = min(T, t + int(chunk_size))
            L = end - t

            X = torch.empty((self.N, L), device=self.device, dtype=self.dtype)

            # Generate states for times t..end-1.
            for j in range(L):
                X[:, j] = x
                tj = t + j
                if tj < T - 1:
                    Ax = torch.sparse.mm(self.A, x.unsqueeze(1)).squeeze(1)
                    x = torch.tanh(Ax + self._input_term(u_train[:, tj]))

            Phi = X
            self._augment_state_inplace(Phi)

            Y_chunk = u_train[:, t:end]
            # S += Phi Phi^T,  YX += Y Phi^T
            S.addmm_(Phi, Phi.T, beta=1.0, alpha=1.0)
            YX.addmm_(Y_chunk, Phi.T, beta=1.0, alpha=1.0)

            t = end

        # Store final state (matches x = states(:,end) in MATLAB).
        self.state = x

        # Ridge regression solution: Wout = YX (S + beta I)^{-1}
        if self.beta > 0:
            S = S + (self.beta * torch.eye(self.N, device=self.device, dtype=self.dtype))

        # Cholesky solve is stable for SPD matrices.
        Lchol = torch.linalg.cholesky(S)
        w_out_T = torch.cholesky_solve(YX.T, Lchol)
        self.w_out = w_out_T.T
        return self

    @torch.no_grad()
    def predict(self, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autonomous prediction (closed-loop), matching MATLAB `predict.m`.

        Args:
            steps: number of prediction steps.

        Returns:
            outputs: shape [Din, steps]
            final_state: shape [N]
        """
        if steps <= 0:
            raise ValueError("steps must be positive")

        outputs = torch.empty((self.num_inputs, int(steps)), device=self.device, dtype=self.dtype)
        x = self.state

        for i in range(int(steps)):
            x_aug = x.clone()
            self._augment_state_inplace(x_aug)
            out = self.w_out @ x_aug
            outputs[:, i] = out
            Ax = torch.sparse.mm(self.A, x.unsqueeze(1)).squeeze(1)
            x = torch.tanh(Ax + self._input_term(out))

        self.state = x
        return outputs, x


@torch.no_grad()
def kursiv_solve_etdrk4(
    init: torch.Tensor,
    *,
    dt: float,
    n_steps: int,
    d: float,
    M: int = 16,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Integrate the standard KS equation using the ETDRK4 scheme.

    This matches `KSBasicSingleReservoir/kursiv_solve.m` (Kassam & Trefethen).

    Args:
        init: shape [N] initial condition in real space.
        dt: time step (h in MATLAB).
        n_steps: number of integration steps.
        d: domain length / periodicity.
        M: number of points for complex means.

    Returns:
        uu: shape [n_steps, N] real-valued field snapshots.
            The first row corresponds to the solution after 1 step (t=dt),
            matching the MATLAB code's `vv(:,n)=v` inside the loop.
    """
    if init.ndim != 1:
        raise ValueError("init must be 1D")
    N = int(init.shape[0])
    if N % 2 != 0:
        raise ValueError("N must be even (as assumed by the MATLAB code)")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if d <= 0:
        raise ValueError("d must be positive")

    dev = _as_device(device)
    init = init.to(device=dev, dtype=dtype)

    # Fourier wave numbers (MATLAB: [0:N/2-1 0 -N/2+1:-1]'*(2*pi/d))
    k_pos = torch.arange(0, N // 2, device=dev, dtype=dtype)
    k_mid = torch.zeros(1, device=dev, dtype=dtype)
    k_neg = torch.arange(-N // 2 + 1, 0, device=dev, dtype=dtype)
    k = torch.cat([k_pos, k_mid, k_neg], dim=0) * (2.0 * math.pi / float(d))

    L = k**2 - k**4
    E = torch.exp(float(dt) * L)
    E2 = torch.exp(float(dt) * L / 2.0)

    # Roots of unity (complex)
    m = torch.arange(1, M + 1, device=dev, dtype=dtype)
    r = torch.exp(1j * math.pi * (m - 0.5) / float(M))
    LR = float(dt) * L.unsqueeze(1) + r.unsqueeze(0)

    Q = float(dt) * torch.real(torch.mean((torch.exp(LR / 2.0) - 1.0) / LR, dim=1))
    f1 = float(dt) * torch.real(
        torch.mean((-4.0 - LR + torch.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), dim=1)
    )
    f2 = float(dt) * torch.real(torch.mean((2.0 + LR + torch.exp(LR) * (-2.0 + LR)) / (LR**3), dim=1))
    f3 = float(dt) * torch.real(
        torch.mean((-4.0 - 3.0 * LR - LR**2 + torch.exp(LR) * (4.0 - LR)) / (LR**3), dim=1)
    )

    g = (-0.5j) * k
    v = torch.fft.fft(init.to(dtype=dtype))  # complex

    uu = torch.empty((int(n_steps), N), device=dev, dtype=dtype)
    for n in range(int(n_steps)):
        Nv = g * torch.fft.fft(torch.real(torch.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * torch.fft.fft(torch.real(torch.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * torch.fft.fft(torch.real(torch.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * torch.fft.fft(torch.real(torch.fft.ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        uu[n] = torch.real(torch.fft.ifft(v))
    return uu


@torch.no_grad()
def reproduce_prl_figure2(
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
    seed_data: Optional[int] = None,
    seed_A: Optional[int] = None,
    # KS parameters (PRL Fig.2: L=22, Q=64, mu=0)
    ks_N: int = 64,
    ks_d: float = 22.0,
    ks_dt: float = 0.25,
    ks_steps: int = 100_000,
    # Reservoir parameters
    approx_reservoir_size: int = 5000,
    radius: float = 0.6,
    degree: float = 3.0,
    sigma: float = 1.0,
    beta: float = 1e-4,
    train_length: int = 70_000,
    predict_length: int = 1_000,
    lambda_max: float = 0.05,
    # Plot settings
    xlim: Tuple[float, float] = (0.0, 12.0),
):
    """Reproduce Fig. 2 style plots (actual / prediction / error) for the KS demo.

    Returns:
        fig, (actual, pred, err), (t_axis, x_axis)
    """
    import matplotlib.pyplot as plt

    dev = _as_device(device)

    # --- KS data ---
    if seed_data is not None:
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed_data))
        init = 0.6 * (-1.0 + 2.0 * torch.rand(ks_N, generator=gen, device=dev, dtype=dtype))
    else:
        init = 0.6 * (-1.0 + 2.0 * torch.rand(ks_N, device=dev, dtype=dtype))

    uu = kursiv_solve_etdrk4(init, dt=ks_dt, n_steps=ks_steps, d=ks_d, device=dev, dtype=dtype)
    data = uu.T  # [N, T]

    if train_length + predict_length > data.shape[1]:
        raise ValueError("Need ks_steps >= train_length + predict_length")

    # --- Reservoir ---
    num_inputs = ks_N
    reservoir_size = (approx_reservoir_size // num_inputs) * num_inputs
    model = KSBasicSingleReservoir(
        num_inputs=num_inputs,
        reservoir_size=reservoir_size,
        radius=radius,
        degree=degree,
        sigma=sigma,
        beta=beta,
        seed_A=seed_A,
        device=dev,
        dtype=dtype,
    )
    model.fit(data[:, :train_length], chunk_size=2048)
    pred, _ = model.predict(predict_length)

    actual = data[:, train_length : train_length + predict_length]
    err = pred - actual  # PRL Fig.2 caption: (b) minus (a)

    # Axes (match MATLAB script / PRL style)
    t_axis = torch.arange(1, predict_length + 1, device=dev, dtype=dtype) * float(ks_dt) * float(lambda_max)
    x_axis = torch.linspace(0.0, float(ks_d), steps=ks_N + 1, device=dev, dtype=dtype)[:-1]

    # --- Plot ---
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), constrained_layout=True)
    extent = [float(t_axis[0]), float(t_axis[-1]), float(x_axis[0]), float(x_axis[-1])]
    im0 = axs[0].imshow(actual.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[0].set_title("Actual")
    axs[0].set_xlabel(r"$\Lambda_{max} t$")
    axs[0].set_ylabel("x")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[1].set_title("Prediction")
    axs[1].set_xlabel(r"$\Lambda_{max} t$")
    axs[1].set_ylabel("x")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(err.cpu().numpy(), aspect="auto", origin="lower", extent=extent)
    axs[2].set_title("Error (prediction - actual)")
    axs[2].set_xlabel(r"$\Lambda_{max} t$")
    axs[2].set_ylabel("x")
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.set_xlim(*xlim)

    # Jet colormap like MATLAB
    plt.set_cmap("jet")

    return fig, (actual, pred, err), (t_axis, x_axis)
