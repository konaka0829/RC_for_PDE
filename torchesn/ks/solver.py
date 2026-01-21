import math
from typing import Optional

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
    """Integrate the standard KS equation using the ETDRK4 scheme.

    This matches the MATLAB ETDRK4 implementation (Kassam & Trefethen).

    Args:
        init: shape [N] initial condition in real space.
        dt: time step (h in MATLAB).
        n_steps: number of integration steps.
        d: domain length / periodicity.
        M: number of points for complex means.
        device: device for computation.
        dtype: tensor dtype for real values.

    Returns:
        uu: shape [n_steps, N] real-valued field snapshots.
            The first row corresponds to the solution after 1 step (t=dt),
            matching the MATLAB code's storage inside the loop.
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
