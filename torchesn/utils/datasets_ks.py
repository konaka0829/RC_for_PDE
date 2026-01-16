import json
from pathlib import Path

import numpy as np

from .kuramoto_sivashinsky import simulate_ks


def generate_or_load_ks_dataset(
    path,
    *,
    L,
    Q,
    dt,
    mu,
    lam,
    total_steps,
    burn_in,
    seed,
    dtype=np.float32,
):
    """Generate or load a cached KS dataset.

    Args:
        path (str or Path): Output .npz path.
        L, Q, dt, mu, lam: KS parameters.
        total_steps (int): Number of steps to return.
        burn_in (int): Burn-in steps before recording.
        seed (int): RNG seed.
        dtype (np.dtype): dtype for storage.

    Returns:
        np.ndarray: KS trajectory of shape (total_steps, Q).
    """
    path = Path(path)
    metadata = {
        "L": float(L),
        "Q": int(Q),
        "dt": float(dt),
        "mu": float(mu),
        "lam": float(lam),
        "total_steps": int(total_steps),
        "burn_in": int(burn_in),
        "seed": int(seed),
        "dtype": str(np.dtype(dtype)),
    }

    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        saved_meta = json.loads(str(loaded["metadata"]))
        if saved_meta != metadata:
            raise ValueError("Metadata mismatch for cached KS dataset.")
        return loaded["data"]

    path.parent.mkdir(parents=True, exist_ok=True)
    data = simulate_ks(
        L=L,
        Q=Q,
        dt=dt,
        n_steps=total_steps,
        mu=mu,
        lam=lam,
        seed=seed,
        burn_in=burn_in,
        dtype=dtype,
    )

    np.savez(path, data=data, metadata=json.dumps(metadata))
    return data
