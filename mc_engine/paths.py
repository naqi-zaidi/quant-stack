"""Shared driver for path-dependent payoffs: simulate GBM paths, apply a per-path payoff, and
return a price with an honest standard error.

Two samplers:

* ``pseudo``: i.i.d. paths; standard error from the sample variance.
* ``sobol``: randomized QMC. Independent scramblings, each a power-of-2 block. The standard
  error comes from the spread of the replicate means, with Student's t degrees of freedom.

The payoff callback receives paths of shape (m, n_steps + 1), column 0 equal to S0, and
returns one discounted value per path. Pricing, control variates and barrier weights all
live in the callback, so this driver contains no finance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.stats import qmc

from mc_engine.brownian import Construction, brownian_motion, sobol_normals
from mc_engine.pricer import MCResult, Seed, _RunningMoments

Sampler = Literal["pseudo", "sobol"]
PerPath = Callable[[np.ndarray], np.ndarray]
PATH_CHUNK = 1 << 14  # paths per block: 16k x (n_steps + 1) doubles stays small


def seed_sequence(seed: Seed) -> np.random.SeedSequence:
    return seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)


def gbm_paths_from_normals(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    q: float,
    z: np.ndarray,
    construction: Construction = "standard",
) -> np.ndarray:
    """Exact GBM on the grid 0, dt, ..., t from normals ``z`` of shape (m, n_steps)."""
    n_steps = z.shape[1]
    w = brownian_motion(z, t, construction)
    grid = np.linspace(0.0, t, n_steps + 1)
    return s0 * np.exp((r - q - 0.5 * sigma**2) * grid + sigma * w)


def estimate(
    per_path: PerPath,
    s0: float,
    r: float,
    sigma: float,
    t: float,
    q: float,
    n_steps: int,
    n_paths: int,
    seed: Seed,
    *,
    sampler: Sampler = "pseudo",
    construction: Construction = "bridge",
    n_replicates: int = 16,
    method: str = "",
) -> MCResult:
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if sampler == "pseudo":
        if n_paths < 2:
            raise ValueError("n_paths must be at least 2")
        rng = np.random.default_rng(seed)
        moments = _RunningMoments()
        left = n_paths
        while left > 0:
            m = min(PATH_CHUNK, left)
            z = rng.standard_normal((m, n_steps))
            moments.update(per_path(gbm_paths_from_normals(s0, r, sigma, t, q, z, construction)))
            left -= m
        return MCResult(moments.mean, moments.std_error(), n_paths, method=method or "pseudo")

    if sampler != "sobol":
        raise ValueError(f"sampler must be 'pseudo' or 'sobol', got {sampler!r}")
    per_rep, rem = divmod(n_paths, n_replicates)
    if n_replicates < 2 or rem or per_rep < 2 or per_rep & (per_rep - 1):
        raise ValueError("Sobol needs n_replicates >= 2 and n_paths / n_replicates a power of 2")
    means = np.empty(n_replicates)
    for i, child in enumerate(seed_sequence(seed).spawn(n_replicates)):
        sob = qmc.Sobol(d=n_steps, scramble=True, rng=np.random.default_rng(child))
        total, left = 0.0, per_rep
        while left > 0:
            m = min(PATH_CHUNK, left)
            z = sobol_normals(sob, m)
            total += float(
                per_path(gbm_paths_from_normals(s0, r, sigma, t, q, z, construction)).sum()
            )
            left -= m
        means[i] = total / per_rep
    return MCResult(
        float(means.mean()),
        float(means.std(ddof=1) / np.sqrt(n_replicates)),
        n_paths,
        method=method or f"sobol_{construction}",
        dof=n_replicates - 1,
    )
