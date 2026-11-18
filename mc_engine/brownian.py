"""Brownian motion on an even time grid, built from standard normals two ways.

Both constructions are linear maps z -> W with the same covariance, Cov(W_s, W_t) = min(s, t),
so they give identical prices with pseudo-random numbers. They differ under quasi-Monte Carlo,
where the first Sobol coordinates are far better distributed than the later ones:

* ``standard``: W_{i+1} = W_i + sqrt(dt) z_i. Every coordinate matters equally, so with 64 steps
  the terminal value depends on all 64 Sobol dimensions.
* ``bridge``: W_T = sqrt(T) z_0 first, then midpoints conditioned on their endpoints,
  breadth-first. The terminal value uses only z_0, and coarse path shape uses the next few
  coordinates, so most of the variance sits in the dimensions Sobol handles best.
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

Construction = Literal["standard", "bridge"]


@cache
def bridge_schedule(n_steps: int) -> tuple[tuple[int, int, int, float, float, float], ...]:
    """Fill order for the bridge on grid indices 0..n_steps, in units where dt = 1.

    Each entry (m, left, right, w_left, w_right, sd) sets
        W[m] = w_left * W[left] + w_right * W[right] + sd * sqrt(dt) * z
    The terminal point (index n_steps) is set separately from z_0.
    """
    schedule = []
    queue = [(0, n_steps)]
    while queue:
        nxt = []
        for left, right in queue:
            if right - left < 2:
                continue
            m = (left + right) // 2
            span = right - left
            schedule.append(
                (
                    m,
                    left,
                    right,
                    (right - m) / span,
                    (m - left) / span,
                    float(np.sqrt((m - left) * (right - m) / span)),
                )
            )
            nxt += [(left, m), (m, right)]
        queue = nxt
    return tuple(schedule)


def brownian_motion(z: np.ndarray, t: float, construction: Construction = "standard") -> np.ndarray:
    """Brownian paths from normals ``z`` of shape (n_paths, n_steps).

    Returns W of shape (n_paths, n_steps + 1) on the grid 0, dt, ..., t, with W[:, 0] = 0.
    """
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    n_paths, n_steps = z.shape
    dt = t / n_steps
    w = np.zeros((n_paths, n_steps + 1))
    if construction == "standard":
        np.cumsum(np.sqrt(dt) * z, axis=1, out=w[:, 1:])
        return w
    if construction != "bridge":
        raise ValueError(f"construction must be 'standard' or 'bridge', got {construction!r}")
    w[:, -1] = np.sqrt(t) * z[:, 0]
    sqrt_dt = np.sqrt(dt)
    for j, (m, left, right, wl, wr, sd) in enumerate(bridge_schedule(n_steps), start=1):
        w[:, m] = wl * w[:, left] + wr * w[:, right] + sd * sqrt_dt * z[:, j]
    return w


def sobol_normals(sampler: qmc.Sobol, n: int) -> np.ndarray:
    """Next ``n`` points of a scrambled Sobol sequence, mapped to N(0, 1) by the inverse CDF.

    ``n`` should be a power of 2 so each draw keeps Sobol's balance properties.
    """
    u = sampler.random(n)
    # Scrambled points lie in [0, 1); guard the measure-zero endpoint before inverting.
    np.clip(u, np.finfo(np.float64).tiny, None, out=u)
    return ndtri(u)
