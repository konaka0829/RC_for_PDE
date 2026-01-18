"""Kuramoto-Sivashinsky solver translated from MATLAB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelParams:
    N: int
    d: float
    tau: float
    nstep: int


def _select_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_k(N: int, d: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = N // 2
    positive = torch.arange(0, half, device=device)
    negative = torch.arange(-half + 1, 0, device=device)
    k = torch.cat([positive, torch.zeros(1, device=device), negative])
    k = k.to(dtype=dtype) * (2.0 * torch.pi / d)
    return k


def kursiv_solve(
    init, params: ModelParams, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    device = _select_device(device)
    N = params.N
    h = params.tau
    nmax = params.nstep

    init_tensor = torch.as_tensor(init, dtype=dtype, device=device)
    u = init_tensor.reshape(-1)

    k = _build_k(N, params.d, device, dtype)
    L = k**2 - k**4
    E = torch.exp(h * L)
    E2 = torch.exp(h * L / 2.0)

    M = 16
    r = torch.exp(1j * torch.pi * (torch.arange(1, M + 1, device=device, dtype=dtype) - 0.5) / M)
    LR = h * L[:, None] + r[None, :]
    Q = h * torch.real(torch.mean((torch.exp(LR / 2.0) - 1.0) / LR, dim=1))
    f1 = h * torch.real(
        torch.mean((-4.0 - LR + torch.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR**3, dim=1)
    )
    f2 = h * torch.real(
        torch.mean((2.0 + LR + torch.exp(LR) * (-2.0 + LR)) / LR**3, dim=1)
    )
    f3 = h * torch.real(
        torch.mean((-4.0 - 3.0 * LR - LR**2 + torch.exp(LR) * (4.0 - LR)) / LR**3, dim=1)
    )

    g = -0.5j * k
    v = torch.fft.fft(u.to(dtype))
    vv = torch.zeros((N, nmax), device=device, dtype=torch.complex128 if dtype == torch.float64 else torch.complex64)

    for n in range(nmax):
        Nv = g * torch.fft.fft(torch.real(torch.fft.ifft(v)) ** 2)
        a = E2 * v + Q * Nv
        Na = g * torch.fft.fft(torch.real(torch.fft.ifft(a)) ** 2)
        b = E2 * v + Q * Na
        Nb = g * torch.fft.fft(torch.real(torch.fft.ifft(b)) ** 2)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = g * torch.fft.fft(torch.real(torch.fft.ifft(c)) ** 2)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        vv[:, n] = v

    uu = torch.real(torch.fft.ifft(vv, dim=0)).T
    return uu.to(dtype)
