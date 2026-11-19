"""Discretely monitored Asian options under GBM.

The average runs over the n monitoring dates t_i = i T / n, i = 1..n (S0 is excluded).

* Geometric average: ln G is normal, so the price has a closed form. Its mean and variance
  follow from Cov(W_s, W_t) = min(s, t):
      E[ln G]   = ln S0 + (r - q - sigma^2 / 2) T (n + 1) / (2n)
      Var[ln G] = sigma^2 T (n + 1)(2n + 1) / (6 n^2)
* Arithmetic average: no closed form. But A and G are almost perfectly correlated, so the
  geometric payoff, whose mean is known exactly, is a near-ideal control variate
  (Kemna and Vorst, 1990).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import ndtr

from mc_engine.black_scholes import OptionKind
from mc_engine.brownian import Construction
from mc_engine.paths import Sampler, estimate, gbm_paths_from_normals, seed_sequence
from mc_engine.pricer import MCResult, Seed, payoff

Average = Literal["arithmetic", "geometric"]


def geometric_asian_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "call",
    n_steps: int = 12,
) -> float:
    """Closed-form price of a discretely monitored geometric-average Asian option."""
    n = n_steps
    mu = np.log(s0) + (r - q - 0.5 * sigma**2) * t * (n + 1) / (2 * n)
    var = sigma**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = np.sqrt(var)
    forward = np.exp(mu + 0.5 * var)  # E[G]
    d1 = (mu - np.log(k) + var) / sd
    d2 = d1 - sd
    disc = np.exp(-r * t)
    if kind == "call":
        return float(disc * (forward * ndtr(d1) - k * ndtr(d2)))
    if kind == "put":
        return float(disc * (k * ndtr(-d2) - forward * ndtr(-d1)))
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _averages(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    monitored = paths[:, 1:]
    return monitored.mean(axis=1), np.exp(np.log(monitored).mean(axis=1))


def price_asian(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "call",
    n_steps: int = 12,
    n_paths: int = 1 << 16,
    seed: Seed = 0,
    *,
    average: Average = "arithmetic",
    control_variate: bool = False,
    sampler: Sampler = "pseudo",
    construction: Construction = "bridge",
    n_replicates: int = 16,
    pilot_paths: int = 10_000,
) -> MCResult:
    """Monte Carlo price of an Asian option, optionally with the geometric control variate.

    With ``control_variate=True`` (arithmetic only), beta comes from an independent pilot run of
    ``pilot_paths`` pseudo-random paths, so the main estimate stays unbiased.
    """
    if average not in ("arithmetic", "geometric"):
        raise ValueError(f"average must be 'arithmetic' or 'geometric', got {average!r}")
    if control_variate and average == "geometric":
        raise ValueError("the geometric control variate applies to arithmetic averages only")
    pilot_seed, main_seed = seed_sequence(seed).spawn(2)
    disc = np.exp(-r * t)

    if average == "geometric":

        def per_path(p):
            return disc * payoff(_averages(p)[1], k, kind)

        name = "geometric"
    elif not control_variate:

        def per_path(p):
            return disc * payoff(_averages(p)[0], k, kind)

        name = "arithmetic"
    else:
        mean_x = geometric_asian_price(s0, k, t, r, sigma, q, kind, n_steps)
        z = np.random.default_rng(pilot_seed).standard_normal((pilot_paths, n_steps))
        a, g = _averages(gbm_paths_from_normals(s0, r, sigma, t, q, z))
        x, y = disc * payoff(g, k, kind), disc * payoff(a, k, kind)
        var_x = x.var(ddof=1)
        beta = float(np.cov(x, y)[0, 1] / var_x) if var_x > 0 else 0.0

        def per_path(p):
            a, g = _averages(p)
            return disc * payoff(a, k, kind) - beta * (disc * payoff(g, k, kind) - mean_x)

        name = "arithmetic_cv"

    return estimate(
        per_path,
        s0,
        r,
        sigma,
        t,
        q,
        n_steps,
        n_paths,
        main_seed,
        sampler=sampler,
        construction=construction,
        n_replicates=n_replicates,
        method=f"{name}_{sampler}",
    )
