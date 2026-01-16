import numpy as np


def simulate_ks(
    L,
    Q,
    dt,
    n_steps,
    mu=0.0,
    lam=100.0,
    seed=0,
    burn_in=0,
    u0=None,
    dtype=np.float32,
    return_x=False,
):
    """Simulate the 1D Kuramoto-Sivashinsky equation with ETDRK4.

    The PDE is:
        u_t = -u u_x - u_xx - u_xxxx + mu * cos(2pi x / lam)

    Args:
        L (float): Domain length.
        Q (int): Number of spatial grid points.
        dt (float): Time step.
        n_steps (int): Number of steps to return (includes initial state).
        mu (float): Forcing amplitude.
        lam (float): Forcing wavelength.
        seed (int): RNG seed used when u0 is None.
        burn_in (int): Number of burn-in steps before collecting output.
        u0 (np.ndarray, optional): Initial condition of shape (Q,).
        dtype (np.dtype): Output dtype.
        return_x (bool): If True, also return spatial grid x.

    Returns:
        np.ndarray or (np.ndarray, np.ndarray): Trajectory of shape (n_steps, Q)
        and optionally x grid of shape (Q,).
    """
    if Q <= 0:
        raise ValueError("Q must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")

    dx = L / Q
    x = L * np.arange(Q, dtype=dtype) / Q
    k = 2.0 * np.pi * np.fft.fftfreq(Q, d=dx)
    Lk = k ** 2 - k ** 4

    if u0 is None:
        rng = np.random.RandomState(seed)
        u = 0.01 * rng.randn(Q).astype(dtype)
    else:
        if u0.shape != (Q,):
            raise ValueError("u0 must have shape (Q,).")
        u = u0.astype(dtype, copy=True)

    f = mu * np.cos(2.0 * np.pi * x / lam)
    f_hat = np.fft.fft(f)

    E = np.exp(dt * Lk)
    E2 = np.exp(dt * Lk / 2.0)

    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, None] + r[None, :]
    Qr = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(
        np.mean((-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1)
    )
    f2 = dt * np.real(
        np.mean((2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1)
    )
    f3 = dt * np.real(
        np.mean((-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1)
    )

    def nonlinear_hat(state):
        return -0.5j * k * np.fft.fft(state ** 2) + f_hat

    def step(state):
        v = np.fft.fft(state)
        Nv = nonlinear_hat(state)
        a = E2 * v + Qr * Nv
        Na = nonlinear_hat(np.fft.ifft(a).real)
        b = E2 * v + Qr * Na
        Nb = nonlinear_hat(np.fft.ifft(b).real)
        c = E2 * a + Qr * (2.0 * Nb - Nv)
        Nc = nonlinear_hat(np.fft.ifft(c).real)
        v_next = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        return np.fft.ifft(v_next).real.astype(dtype)

    for _ in range(burn_in):
        u = step(u)

    trajectory = np.zeros((n_steps, Q), dtype=dtype)
    trajectory[0] = u
    for t in range(1, n_steps):
        u = step(u)
        trajectory[t] = u

    if return_x:
        return trajectory, x
    return trajectory
