"""KS equation solvers and dataset helpers."""

from __future__ import annotations

from typing import Optional

import math

import torch


def _as_device(device: Optional[torch.device | str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


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
    """Integrate the standard KS equation using the ETDRK4 scheme."""
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

    k_pos = torch.arange(0, N // 2, device=dev, dtype=dtype)
    k_mid = torch.zeros(1, device=dev, dtype=dtype)
    k_neg = torch.arange(-N // 2 + 1, 0, device=dev, dtype=dtype)
    k = torch.cat([k_pos, k_mid, k_neg], dim=0) * (2.0 * math.pi / float(d))

    L = k**2 - k**4
    E = torch.exp(float(dt) * L)
    E2 = torch.exp(float(dt) * L / 2.0)

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
    v = torch.fft.fft(init.to(dtype=dtype))

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
def ks_solve_etdrk4_forced(
    init: torch.Tensor,
    *,
    dt: float,
    n_steps: int,
    L: float,
    mu: float = 0.0,
    wavelength: float = 100.0,
    M: int = 16,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Integrate the *forced* KS equation with ETDRK4."""
    if init.ndim != 1:
        raise ValueError("init must be 1D")
    Q = int(init.shape[0])
    if Q % 2 != 0:
        raise ValueError("Q must be even (as assumed by the ETDRK4 KS scheme)")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if M <= 0:
        raise ValueError("M must be positive")

    dev = _as_device(device)
    init = init.to(device=dev, dtype=dtype)

    k_pos = torch.arange(0, Q // 2, device=dev, dtype=dtype)
    k_mid = torch.zeros(1, device=dev, dtype=dtype)
    k_neg = torch.arange(-Q // 2 + 1, 0, device=dev, dtype=dtype)
    k = torch.cat([k_pos, k_mid, k_neg], dim=0) * (2.0 * math.pi / float(L))

    lin = k**2 - k**4
    E = torch.exp(float(dt) * lin)
    E2 = torch.exp(float(dt) * lin / 2.0)

    m = torch.arange(1, int(M) + 1, device=dev, dtype=dtype)
    r = torch.exp(1j * math.pi * (m - 0.5) / float(M))
    LR = float(dt) * lin.unsqueeze(1) + r.unsqueeze(0)

    Qc = float(dt) * torch.real(torch.mean((torch.exp(LR / 2.0) - 1.0) / LR, dim=1))
    f1 = float(dt) * torch.real(
        torch.mean((-4.0 - LR + torch.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / (LR**3), dim=1)
    )
    f2 = float(dt) * torch.real(torch.mean((2.0 + LR + torch.exp(LR) * (-2.0 + LR)) / (LR**3), dim=1))
    f3 = float(dt) * torch.real(
        torch.mean((-4.0 - 3.0 * LR - LR**2 + torch.exp(LR) * (4.0 - LR)) / (LR**3), dim=1)
    )

    g = (-0.5j) * k
    v = torch.fft.fft(init)

    forcing_hat = None
    if float(mu) != 0.0:
        x = torch.arange(Q, device=dev, dtype=dtype) * (float(L) / float(Q))
        forcing = float(mu) * torch.cos(2.0 * math.pi * x / float(wavelength))
        forcing_hat = torch.fft.fft(forcing)

    uu = torch.empty((int(n_steps), Q), device=dev, dtype=dtype)
    for n in range(int(n_steps)):
        u = torch.real(torch.fft.ifft(v))
        Nv = g * torch.fft.fft(u**2)
        if forcing_hat is not None:
            Nv = Nv + forcing_hat

        a = E2 * v + Qc * Nv
        ua = torch.real(torch.fft.ifft(a))
        Na = g * torch.fft.fft(ua**2)
        if forcing_hat is not None:
            Na = Na + forcing_hat

        b = E2 * v + Qc * Na
        ub = torch.real(torch.fft.ifft(b))
        Nb = g * torch.fft.fft(ub**2)
        if forcing_hat is not None:
            Nb = Nb + forcing_hat

        c = E2 * a + Qc * (2.0 * Nb - Nv)
        uc = torch.real(torch.fft.ifft(c))
        Nc = g * torch.fft.fft(uc**2)
        if forcing_hat is not None:
            Nc = Nc + forcing_hat

        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        uu[n] = torch.real(torch.fft.ifft(v))

    return uu


@torch.no_grad()
def generate_ks_dataset(
    *,
    Q: int,
    L: float,
    mu: float,
    wavelength: float,
    dt: float,
    n_steps: int,
    burn_in: int = 0,
    seed: Optional[int] = None,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate a KS dataset suitable for training and evaluation."""
    if Q <= 0 or Q % 2 != 0:
        raise ValueError("Q must be a positive even integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be >= 0")

    dev = _as_device(device)
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))

    init = 0.6 * (2.0 * torch.rand(Q, generator=gen, device=dev, dtype=dtype) - 1.0)

    total = int(burn_in + n_steps)
    uu = ks_solve_etdrk4_forced(
        init,
        dt=float(dt),
        n_steps=total,
        L=float(L),
        mu=float(mu),
        wavelength=float(wavelength),
        device=dev,
        dtype=dtype,
    )
    if burn_in > 0:
        uu = uu[int(burn_in) :]
    return uu.T.contiguous()
