"""PyTorch implementation matching KSBasicSingleReservoir MATLAB behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch


@dataclass
class ReservoirParams:
    N: int
    radius: float
    degree: float
    sigma: float
    train_length: int
    num_inputs: int
    predict_length: int
    beta: float


def generate_reservoir_scipy(
    size: int, radius: float, degree: float, seed: Optional[int] = None
) -> scipy.sparse.csr_matrix:
    sparsity = degree / size
    rng = np.random.RandomState(seed) if seed is not None else None
    A = scipy.sparse.random(
        size,
        size,
        density=sparsity,
        format="csr",
        random_state=rng,
        data_rvs=None,
    )

    if size <= 2:
        eigvals = np.linalg.eigvals(A.toarray())
        e = np.max(np.abs(eigvals))
    else:
        k = min(6, size - 2)
        try:
            eigvals = scipy.sparse.linalg.eigs(
                A, k=k, which="LM", return_eigenvectors=False
            )
            e = np.max(np.abs(eigvals))
        except Exception:
            eigvals = np.linalg.eigvals(A.toarray())
            e = np.max(np.abs(eigvals))

    if e == 0:
        return A

    A = (A / e) * radius
    return A.tocsr()


def scipy_sparse_to_torch(
    A_csr: scipy.sparse.csr_matrix,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    A_coo = A_csr.tocoo()
    indices = torch.tensor([A_coo.row, A_coo.col], dtype=torch.long, device=device)
    values = torch.tensor(A_coo.data, dtype=dtype, device=device)
    A_torch = torch.sparse_coo_tensor(indices, values, A_coo.shape, device=device, dtype=dtype)
    return A_torch.coalesce()


def build_win(
    params: ReservoirParams, device: torch.device, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    q = params.N // params.num_inputs
    win = torch.zeros((params.N, params.num_inputs), device=device, dtype=dtype)
    for i in range(1, params.num_inputs + 1):
        rng = np.random.RandomState(i)
        ip = params.sigma * (-1.0 + 2.0 * rng.rand(q, 1))
        win[(i - 1) * q : i * q, i - 1] = torch.as_tensor(ip[:, 0], dtype=dtype, device=device)
    return win


def reservoir_layer(
    A_sparse: torch.Tensor, win: torch.Tensor, data: torch.Tensor, params: ReservoirParams
) -> torch.Tensor:
    states = torch.zeros((params.N, params.train_length), device=data.device, dtype=data.dtype)
    for i in range(params.train_length - 1):
        prev = states[:, i].unsqueeze(1)
        ax = torch.sparse.mm(A_sparse, prev).squeeze(1)
        ux = win @ data[:, i]
        states[:, i + 1] = torch.tanh(ax + ux)
    return states


def augment_even_square(x_or_states: torch.Tensor) -> torch.Tensor:
    augmented = x_or_states.clone()
    if augmented.ndim == 1:
        augmented[1::2] = augmented[1::2] ** 2
    elif augmented.ndim == 2:
        augmented[1::2, :] = augmented[1::2, :] ** 2
    else:
        raise ValueError("Input must be 1D or 2D tensor.")
    return augmented


def train_wout(
    params: ReservoirParams,
    states: torch.Tensor,
    data: torch.Tensor,
    method: str = "pinv",
) -> torch.Tensor:
    if method != "pinv":
        raise ValueError("Only pinv method is supported.")
    states_aug = augment_even_square(states)
    idenmat = params.beta * torch.eye(params.N, device=states.device, dtype=states.dtype)
    gram = states_aug @ states_aug.T + idenmat
    w_out = data @ states_aug.T @ torch.linalg.pinv(gram)
    return w_out


def _select_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_reservoir(
    params: ReservoirParams,
    data: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    seed_A: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _select_device(device)
    data_tensor = data.to(device=device, dtype=dtype)
    A_csr = generate_reservoir_scipy(params.N, params.radius, params.degree, seed=seed_A)
    A_sparse = scipy_sparse_to_torch(A_csr, device=device, dtype=dtype)
    win = build_win(params, device=device, dtype=dtype)
    states = reservoir_layer(A_sparse, win, data_tensor, params)
    wout = train_wout(params, states, data_tensor)
    x_last = states[:, -1]
    return x_last, wout, A_sparse, win


def predict(
    A_sparse: torch.Tensor,
    win: torch.Tensor,
    params: ReservoirParams,
    x: torch.Tensor,
    wout: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros(
        (params.num_inputs, params.predict_length), device=x.device, dtype=x.dtype
    )
    x_state = x.view(-1)
    for i in range(params.predict_length):
        x_aug = augment_even_square(x_state)
        out = wout @ x_aug
        output[:, i] = out
        ax = torch.sparse.mm(A_sparse, x_state.unsqueeze(1)).squeeze(1)
        x_state = torch.tanh(ax + win @ out)
    return output, x_state


def train_reservoir_streaming(
    params: ReservoirParams,
    data: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    seed_A: Optional[int] = None,
    block_size: int = 1024,
    method: str = "pinv",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _select_device(device)
    data_tensor = data.to(device=device, dtype=dtype)
    A_csr = generate_reservoir_scipy(params.N, params.radius, params.degree, seed=seed_A)
    A_sparse = scipy_sparse_to_torch(A_csr, device=device, dtype=dtype)
    win = build_win(params, device=device, dtype=dtype)

    S = torch.zeros((params.N, params.N), device=device, dtype=dtype)
    D = torch.zeros((params.num_inputs, params.N), device=device, dtype=dtype)

    x = torch.zeros(params.N, device=device, dtype=dtype)
    x_block = []
    y_block = []

    for t in range(params.train_length):
        x_block.append(augment_even_square(x))
        y_block.append(data_tensor[:, t])

        if len(x_block) == block_size or t == params.train_length - 1:
            X_block = torch.stack(x_block, dim=1)
            Y_block = torch.stack(y_block, dim=1)
            S = S + X_block @ X_block.T
            D = D + Y_block @ X_block.T
            x_block.clear()
            y_block.clear()

        if t < params.train_length - 1:
            ax = torch.sparse.mm(A_sparse, x.unsqueeze(1)).squeeze(1)
            x = torch.tanh(ax + win @ data_tensor[:, t])

    x_last = x
    idenmat = params.beta * torch.eye(params.N, device=device, dtype=dtype)
    gram = S + idenmat
    if method == "pinv":
        wout = D @ torch.linalg.pinv(gram)
    elif method == "solve":
        wout = torch.linalg.solve(gram, D.T).T
    else:
        raise ValueError("Only pinv or solve methods are supported.")

    return x_last, wout, A_sparse, win
