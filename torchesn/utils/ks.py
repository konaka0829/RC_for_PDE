import torch


def simulate_ks(L, Q, dt, mu, lam, n_steps, u0=None, seed=None):
    """Simulate a 1D Kuramoto–Sivashinsky-like system with periodic boundaries.

    Args:
        L (float): Domain length.
        Q (int): Number of spatial grid points.
        dt (float): Time step.
        mu (float): Linear growth coefficient.
        lam (float): Constant forcing term.
        n_steps (int): Number of integration steps.
        u0 (Tensor, optional): Initial state of shape (Q,).
        seed (int, optional): Random seed for initialization when u0 is None.

    Returns:
        Tensor: Trajectory of shape (n_steps + 1, Q).
    """
    if Q <= 0:
        raise ValueError("Q must be a positive integer.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if u0 is not None:
        u = u0.clone()
    else:
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.randn(Q)

    if u.dim() != 1 or u.size(0) != Q:
        raise ValueError("u0 must have shape (Q,).")

    dx = L / Q
    trajectory = [u.clone()]
    for _ in range(n_steps):
        u_x = (torch.roll(u, -1) - torch.roll(u, 1)) / (2 * dx)
        u_xx = (torch.roll(u, -1) - 2 * u + torch.roll(u, 1)) / (dx ** 2)
        u_xxxx = (
            torch.roll(u, -2)
            - 4 * torch.roll(u, -1)
            + 6 * u
            - 4 * torch.roll(u, 1)
            + torch.roll(u, 2)
        ) / (dx ** 4)
        du = -u * u_x - u_xx - u_xxxx + mu * u + lam
        u = u + dt * du
        trajectory.append(u.clone())

    return torch.stack(trajectory, dim=0)
