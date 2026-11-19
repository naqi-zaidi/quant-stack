"""Continuously monitored down-and-out call under GBM: closed form and two Monte Carlo
estimators.

A simulated path is only observed on its grid. Between grid points it can dip below the
barrier and come back up unseen, so naive discrete monitoring misses knock-outs and
overprices the option. The bias shrinks only like 1/sqrt(n_steps) (Broadie, Glasserman and
Kou, 1997).

The Brownian-bridge correction removes the bias without a finer grid. ln S is a Brownian motion
with drift. Conditional on its values at two grid points, the chance it crossed ln B in
between is exactly
    p_i = exp(-2 ln(S_i / B) ln(S_{i+1} / B) / (sigma^2 dt))   when both are above B,
whatever the drift. Each path is weighted by its survival probability, prod(1 - p_i), instead
of a hard 0/1. The estimator is unbiased for continuous monitoring at any n_steps, and lower
variance, because it averages over the unseen in-between paths.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from mc_engine.black_scholes import bs_price
from mc_engine.brownian import Construction
from mc_engine.paths import Sampler, estimate
from mc_engine.pricer import MCResult, Seed


def down_and_out_call_price(
    s0: float, k: float, b: float, t: float, r: float, sigma: float, q: float = 0.0
) -> float:
    """Closed form for continuous monitoring, barrier B at or below strike K, no rebate.

    c_do = c - c_di with
        c_di = S0 e^{-qT} (B/S0)^{2 lam} N(y) - K e^{-rT} (B/S0)^{2 lam - 2} N(y - sigma sqrt T)
        lam  = (r - q + sigma^2 / 2) / sigma^2
        y    = ln(B^2 / (S0 K)) / (sigma sqrt T) + lam sigma sqrt T
    (Merton 1973; Reiner and Rubinstein 1991; Hull, Options, Futures and Other Derivatives.)
    """
    if not 0 < b < s0:
        raise ValueError("barrier must be positive and below spot")
    if b > k:
        raise ValueError("this closed form covers B <= K only")
    vol_t = sigma * np.sqrt(t)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(b**2 / (s0 * k)) / vol_t + lam * vol_t
    c_di = s0 * np.exp(-q * t) * (b / s0) ** (2 * lam) * ndtr(y) - k * np.exp(-r * t) * (
        b / s0
    ) ** (2 * lam - 2) * ndtr(y - vol_t)
    return float(bs_price(s0, k, t, r, sigma, q, "call") - c_di)


def price_down_and_out_call(
    s0: float,
    k: float,
    b: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_steps: int = 64,
    n_paths: int = 1 << 16,
    seed: Seed = 0,
    *,
    bridge_correction: bool = True,
    sampler: Sampler = "pseudo",
    construction: Construction = "bridge",
    n_replicates: int = 16,
) -> MCResult:
    """Monte Carlo price for continuous monitoring, simulated on ``n_steps`` grid points.

    ``bridge_correction=False`` is the naive estimator: knock-outs are checked on the grid only.
    """
    if not 0 < b < s0:
        raise ValueError("barrier must be positive and below spot")
    disc = np.exp(-r * t)
    dt = t / n_steps

    def per_path(paths: np.ndarray) -> np.ndarray:
        x = np.log(paths / b)  # distance above the barrier in log space
        alive = np.all(x > 0, axis=1)
        value = disc * np.maximum(paths[:, -1] - k, 0.0) * alive
        if not bridge_correction:
            return value
        xp = np.maximum(x, 0.0)  # dead paths are zeroed by `alive`; avoid overflow noise
        cross = np.exp(-2.0 * xp[:, :-1] * xp[:, 1:] / (sigma**2 * dt))
        return value * np.prod(1.0 - cross, axis=1)

    name = "bridge_corrected" if bridge_correction else "naive"
    return estimate(
        per_path,
        s0,
        r,
        sigma,
        t,
        q,
        n_steps,
        n_paths,
        seed,
        sampler=sampler,
        construction=construction,
        n_replicates=n_replicates,
        method=name,
    )
