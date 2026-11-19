"""Merton (1976) jump-diffusion: GBM plus Poisson jumps with lognormal sizes.

    dS / S = (r - q - lam * kbar) dt + sigma dW + (J - 1) dN,   ln J ~ N(mu_j, delta^2)

Jumps arrive at rate lam. kbar = E[J - 1] = exp(mu_j + delta^2 / 2) - 1, and subtracting
lam * kbar from the drift keeps e^{-(r-q)t} S_t a martingale.

Closed form: conditional on n jumps, ln S_T is normal, so the price is a Poisson-weighted sum
of Black–Scholes prices,
    V = sum_n e^{-lam' T} (lam' T)^n / n! * BS(S0, K, T, r_n, sigma_n, q),
    lam' = lam (1 + kbar),   sigma_n^2 = sigma^2 + n delta^2 / T,
    r_n  = r - lam kbar + n ln(1 + kbar) / T.

Simulation is exact. N_T ~ Poisson(lam T), and the sum of N_T jump sizes is
N_T mu_j + delta sqrt(N_T) Z', so the terminal value needs no time grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from mc_engine.black_scholes import OptionKind, bs_price
from mc_engine.pricer import DEFAULT_CHUNK, MCResult, Seed, _RunningMoments, payoff


@dataclass(frozen=True)
class MertonParams:
    sigma: float  # diffusion volatility
    lam: float  # jump intensity per year
    mu_j: float  # mean log jump size
    delta: float  # sd of log jump size

    @property
    def kbar(self) -> float:
        return float(np.exp(self.mu_j + 0.5 * self.delta**2) - 1.0)


def merton_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    p: MertonParams,
    q: float = 0.0,
    kind: OptionKind = "call",
    tail: float = 1e-16,
) -> float:
    """Series closed form, truncated once the remaining Poisson mass is below ``tail``."""
    lam_p = p.lam * (1.0 + p.kbar)
    n_max = int(poisson.isf(tail, lam_p * t)) + 1 if lam_p > 0 else 0
    n = np.arange(n_max + 1)
    weights = poisson.pmf(n, lam_p * t)
    sigma_n = np.sqrt(p.sigma**2 + n * p.delta**2 / t)
    r_n = r - p.lam * p.kbar + n * np.log1p(p.kbar) / t
    # Each term discounts at its own r_n. That is correct: under the Poisson weights,
    # sum_n w_n e^{-r_n T} = e^{-(r - lam kbar) T} E[(1 + kbar)^{-N}] = e^{-rT}.
    prices = bs_price(s0, k, t, r_n, sigma_n, q, kind)
    return float(np.sum(weights * prices))


def merton_terminal(
    s0: float, t: float, r: float, p: MertonParams, q: float, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Exact draws of S_T."""
    z = rng.standard_normal(n)
    jumps = rng.poisson(p.lam * t, n)
    jump_sum = jumps * p.mu_j + p.delta * np.sqrt(jumps) * rng.standard_normal(n)
    drift = (r - q - p.lam * p.kbar - 0.5 * p.sigma**2) * t
    return s0 * np.exp(drift + p.sigma * np.sqrt(t) * z + jump_sum)


def price_merton_mc(
    s0: float,
    k: float,
    t: float,
    r: float,
    p: MertonParams,
    q: float = 0.0,
    kind: OptionKind = "call",
    n_paths: int = 1 << 18,
    seed: Seed = 0,
    chunk: int = DEFAULT_CHUNK,
) -> MCResult:
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * t)
    moments = _RunningMoments()
    left = n_paths
    while left > 0:
        m = min(chunk, left)
        moments.update(disc * payoff(merton_terminal(s0, t, r, p, q, rng, m), k, kind))
        left -= m
    return MCResult(moments.mean, moments.std_error(), n_paths, method="merton_exact")
