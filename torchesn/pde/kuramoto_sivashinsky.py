"""Kuramoto–Sivashinsky equation utilities."""

from __future__ import annotations

import numpy as np


def simulate_ks_etdrk4(
    *,
    n_steps: int,
    dt: float = 0.25,
    n_grid: int = 64,
    domain_length: float = 22.0,
    transient: int = 0,
    save_every: int = 1,
    seed: int | None = 0,
    u0: "np.ndarray | None" = None,
    mu: float = 0.0,
    forcing_wavelength: float | None = None,
    dealias: bool = False,
    check_interval: int = 200,
    divergence_threshold: float = 1e6,
    dtype: "np.dtype" = np.float64,
) -> dict:
    """Simulate 1D periodic KS equation using ETDRK4."""

    if n_grid % 2 != 0:
        raise ValueError("n_grid must be even for the ETDRK4 scheme.")
    if forcing_wavelength is None:
        if mu != 0.0:
            raise ValueError("forcing_wavelength is required when mu is non-zero.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if save_every <= 0:
        raise ValueError("save_every must be positive.")
    if transient < 0:
        raise ValueError("transient must be non-negative.")

    x = domain_length * np.arange(n_grid, dtype=dtype) / n_grid
    n_half = n_grid // 2
    n_modes = np.concatenate(
        [
            np.arange(0, n_half, dtype=int),
            np.array([0], dtype=int),
            -np.arange(n_half - 1, 0, -1, dtype=int),
        ]
    )
    k = (2 * np.pi / domain_length) * n_modes.astype(dtype)

    L = k**2 - k**4
    E = np.exp(dt * L)
    E2 = np.exp(0.5 * dt * L)

    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(
        np.mean(
            (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3,
            axis=1,
        )
    )
    f2 = dt * np.real(
        np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1)
    )
    f3 = dt * np.real(
        np.mean(
            (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3,
            axis=1,
        )
    )

    if forcing_wavelength is None:
        forcing_hat = np.zeros(n_grid, dtype=np.complex128)
    else:
        forcing = mu * np.cos(2 * np.pi * x / forcing_wavelength)
        forcing_hat = np.fft.fft(forcing)

    if u0 is None:
        rng = np.random.default_rng(seed)
        u = 0.6 * (2 * rng.random(n_grid) - 1)
    else:
        u = np.asarray(u0, dtype=dtype)
        if u.shape != (n_grid,):
            raise ValueError("u0 must have shape (n_grid,).")

    v = np.fft.fft(u)
    v[n_half] = 0.0
    g = -0.5j * k

    def nonlinear_from_u(u_real: np.ndarray) -> np.ndarray:
        u2_hat = np.fft.fft(u_real * u_real)
        if dealias:
            mask = np.abs(n_modes) > n_grid // 3
            u2_hat[mask] = 0.0
        return g * u2_hat + forcing_hat

    def nonlinear(v_hat: np.ndarray) -> np.ndarray:
        u_real = np.fft.ifft(v_hat).real
        return nonlinear_from_u(u_real)

    total_steps = transient + n_steps * save_every
    u_saved = np.empty((n_steps, n_grid), dtype=dtype)
    t_saved = np.empty(n_steps, dtype=dtype)
    save_idx = 0

    def should_check(idx: int, step_idx: int) -> bool:
        if idx == 0 or step_idx == total_steps:
            return True
        if check_interval <= 0:
            return True
        return (idx % check_interval) == 0

    for step in range(1, total_steps + 1):
        u_real = np.fft.ifft(v).real
        # Keep the mean mode anchored to avoid numerical drift.
        u_real -= u_real.mean()
        v = np.fft.fft(u_real)
        Nv = nonlinear_from_u(u_real)
        a = E2 * v + Q * Nv
        Na = nonlinear(a)
        b = E2 * v + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        v[n_half] = 0.0

        if step > transient and (step - transient) % save_every == 0:
            u_real = np.fft.ifft(v).real
            u_saved[save_idx] = u_real
            t_saved[save_idx] = dt * step
            if should_check(save_idx, step):
                if not np.isfinite(u_real).all():
                    raise RuntimeError(
                        f"Non-finite values detected at step {step}."
                    )
                max_abs = float(np.max(np.abs(u_real)))
                if max_abs >= divergence_threshold:
                    raise RuntimeError(
                        "Divergence detected at step "
                        f"{step} with max |u|={max_abs:.3e}."
                    )
            save_idx += 1

    meta = {
        "n_steps": n_steps,
        "dt": dt,
        "n_grid": n_grid,
        "domain_length": domain_length,
        "transient": transient,
        "save_every": save_every,
        "seed": seed,
        "mu": mu,
        "forcing_wavelength": forcing_wavelength,
        "dealias": dealias,
        "check_interval": check_interval,
        "divergence_threshold": divergence_threshold,
        "dtype": str(np.dtype(dtype)),
    }

    return {"u": u_saved, "x": x, "t": t_saved, "meta": meta}
