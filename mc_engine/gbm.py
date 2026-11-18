"""Geometric Brownian motion under the risk-neutral measure, simulated with the exact log step.

    S_{t+dt} = S_t * exp((r - q - sigma^2 / 2) dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)

The -sigma^2/2 term is Itô's correction: ln S has drift r - q - sigma^2/2, which makes
e^{-(r-q)t} S_t a martingale. Because the step is exact there is no time-discretisation
bias, so a European payoff needs only the terminal draw.
"""

from __future__ import annotations

import numpy as np


def _check(s0: float, sigma: float, t: float) -> None:
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if t <= 0:
        raise ValueError("t must be positive")


def terminal(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    z: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """S_T for each standard normal draw in ``z`` (one exact step from 0 to T)."""
    _check(s0, sigma, t)
    return s0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)


def paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    q: float = 0.0,
) -> np.ndarray:
    """Full paths on an even grid, shape (n_paths, n_steps + 1), column 0 equal to s0.

    Needed for path-dependent payoffs (Asian, barrier, American) in later steps.
    """
    _check(s0, sigma, t)
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be at least 1")
    dt = t / n_steps
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
        (n_paths, n_steps)
    )
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return s0 * np.exp(log_paths)
