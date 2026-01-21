from __future__ import annotations

from typing import Optional

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
    """Estimate |lambda_max| for a real sparse matrix using power iteration.

    This mirrors MATLAB's `max(abs(eigs(A)))` for sparse matrices.

    Args:
        A: Square sparse COO tensor (shape [N, N]). Must be coalesced.
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
        Sparse COO tensor A of shape [size, size] with estimated spectral radius `radius`.
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
