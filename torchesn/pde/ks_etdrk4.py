"""Kuramoto–Sivashinsky equation solver using ETDRK4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KSSimulationConfig:
    L: float
    Q: int
    dt: float
    n_steps: int
    mu: float
    lambda_: float
    seed: Optional[int] = None
    u0_scale: float = 0.6
    dtype: np.dtype = np.float32


def _etdrk4_coefficients(L: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    """Compute ETDRK4 scalar coefficients per Kassam & Trefethen (2005)."""
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)

    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    LR3 = LR**3

    Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(
        np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR**2)) / LR3,
            axis=1,
        )
    )
    f2 = dt * np.real(
        np.mean(
            (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR3,
            axis=1,
        )
    )
    f3 = dt * np.real(
        np.mean(
            (-4.0 - 3.0 * LR - LR**2 + np.exp(LR) * (4.0 - LR)) / LR3,
            axis=1,
        )
    )

    return E, E2, Q, f1, f2, f3


def simulate_ks_etdrk4(
    L: float,
    Q: int,
    dt: float,
    n_steps: int,
    mu: float,
    lambda_: float,
    seed: Optional[int] = None,
    u0_scale: float = 0.6,
    dtype: np.dtype | str = np.float32,
    u0: Optional[np.ndarray] = None,
    n_substeps: int = 1,
    filter_strength: float = 0.0,
    filter_order: int = 36,
) -> np.ndarray:
    """Simulate the KS equation using ETDRK4.

    Args:
        L: Domain length.
        Q: Number of spatial grid points.
        dt: Time step.
        n_steps: Number of time steps to simulate.
        mu: Forcing amplitude.
        lambda_: Forcing wavelength.
        seed: Random seed for the initial condition.
        u0_scale: Scale for uniform random initial condition.
        dtype: Output dtype (float32 or float64 recommended).
        u0: Optional initial condition of shape (Q,).

    Returns:
        Array with shape (n_steps, Q) of the simulated field values.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("dtype must be float32 or float64")

    calc_dtype = np.float64 if dtype == np.float32 else dtype
    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")

    complex_dtype = np.complex64 if calc_dtype == np.float32 else np.complex128
    dt_sub = dt / n_substeps

    x = (L / Q) * np.arange(Q, dtype=calc_dtype)
    dx = L / Q
    k = 2.0 * np.pi * np.fft.rfftfreq(Q, d=dx).astype(calc_dtype)
    if Q % 2 == 0:
        k[-1] = 0.0
    mode_indices = np.arange(Q // 2 + 1, dtype=calc_dtype)
    cutoff = Q // 3
    dealias_mask = (mode_indices <= cutoff).astype(complex_dtype)
    spectral_filter = np.ones(Q // 2 + 1, dtype=complex_dtype)
    if filter_strength > 0.0:
        k_abs = np.abs(k)
        k_max = np.max(k_abs) if np.max(k_abs) > 0 else 1.0
        spectral_filter = np.exp(-filter_strength * (k_abs / k_max) ** filter_order).astype(complex_dtype)
    combined_filter = dealias_mask * spectral_filter

    L_hat = k**2 - k**4
    E, E2, Qcoef, f1, f2, f3 = _etdrk4_coefficients(L_hat.astype(complex_dtype), dt_sub)

    forcing = mu * np.cos(2.0 * np.pi * x / lambda_)
    forcing_hat = np.fft.rfft(forcing.astype(calc_dtype)).astype(complex_dtype) * dealias_mask

    if u0 is None:
        rng = np.random.default_rng(seed)
        u = u0_scale * (-1.0 + 2.0 * rng.random(Q).astype(calc_dtype))
    else:
        u = np.asarray(u0, dtype=calc_dtype)
        if u.shape != (Q,):
            raise ValueError("u0 must have shape (Q,)")

    g = -0.5j * k
    v = np.fft.rfft(u).astype(complex_dtype) * combined_filter

    out = np.empty((n_steps, Q), dtype=dtype)

    for step in range(n_steps):
        if not np.isfinite(u).all():
            raise FloatingPointError(
                "Non-finite values detected at step "
                f"{step}. Consider increasing substeps, using float64, or reducing dt."
            )
        out[step] = u.astype(dtype)

        for substep in range(n_substeps):
            Nv = g * np.fft.rfft(u * u).astype(complex_dtype) * dealias_mask + forcing_hat
            a = E2 * v + Qcoef * Nv
            ua = np.fft.irfft(a, n=Q).astype(calc_dtype)
            Na = g * np.fft.rfft(ua * ua).astype(complex_dtype) * dealias_mask + forcing_hat
            b = E2 * v + Qcoef * Na
            ub = np.fft.irfft(b, n=Q).astype(calc_dtype)
            Nb = g * np.fft.rfft(ub * ub).astype(complex_dtype) * dealias_mask + forcing_hat
            c = E2 * a + Qcoef * (2.0 * Nb - Nv)
            uc = np.fft.irfft(c, n=Q).astype(calc_dtype)
            Nc = g * np.fft.rfft(uc * uc).astype(complex_dtype) * dealias_mask + forcing_hat
            v = (E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc) * combined_filter
            u = np.fft.irfft(v, n=Q).astype(calc_dtype)
            if not np.isfinite(u).all():
                raise FloatingPointError(
                    "Non-finite values detected at step "
                    f"{step}, substep {substep}. Consider increasing substeps, "
                    "using float64, or reducing dt."
                )

    return out
