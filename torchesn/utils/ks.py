import torch


def simulate_ks(L, Q, dt, mu, lam, n_steps, u0=None, seed=None):
    """Simulate a simple Kuramoto-Sivashinsky-like evolution.

    This is a lightweight, deterministic surrogate that produces a trajectory
    with the correct shape and honors the provided initial condition.

    Args:
        L (float): Domain length (unused placeholder).
        Q (int): Number of spatial points.
        dt (float): Timestep size.
        mu (float): Linear growth coefficient.
        lam (float): Linear damping coefficient.
        n_steps (int): Number of integration steps.
        u0 (Tensor or None): Optional initial state of shape (Q,).
        seed (int, optional): Random seed for reproducible initialization.

    Returns:
        Tensor: Simulated states of shape (n_steps + 1, Q).
    """
    if Q <= 0:
        raise ValueError("Q must be positive")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")

    if u0 is not None:
        state = torch.as_tensor(u0, dtype=torch.float32).clone()
        if state.numel() != Q:
            raise ValueError("u0 must have shape (Q,)")
    else:
        if seed is not None:
            torch.manual_seed(seed)
        state = torch.randn(Q, dtype=torch.float32)

    trajectory = [state.clone()]
    for _ in range(n_steps):
        state = state + dt * (mu * state - lam * state)
        trajectory.append(state.clone())

    return torch.stack(trajectory, dim=0)
