"""Monte Carlo pricing of European options with a reported standard error.

Paths are drawn in fixed-size chunks and combined with Chan's parallel mean/variance update,
so memory stays flat at any path count and the result is identical to a single large draw
from the same generator.

``price_european(method=...)`` is the single entry point. ``"plain"`` lives here; the
variance-reduced estimators live in ``mc_engine.variance_reduction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from mc_engine import gbm
from mc_engine.black_scholes import OptionKind

DEFAULT_CHUNK = 1 << 20

Method = Literal["plain", "antithetic", "control_variate", "sobol"]
Seed = int | np.random.SeedSequence | None


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int  # payoff evaluations, the cost unit every method is compared on
    method: str = "plain"
    # Degrees of freedom behind the standard error. None means many (normal quantiles);
    # randomized QMC estimates its error from a handful of replicates, so it uses Student's t.
    dof: int | None = None

    def quantile(self, level: float) -> float:
        """Two-sided critical value for a ``level`` confidence interval."""
        p = 0.5 + level / 2
        return float(stats.norm.ppf(p) if self.dof is None else stats.t.ppf(p, self.dof))

    @property
    def ci95(self) -> tuple[float, float]:
        half = self.quantile(0.95) * self.std_error
        return self.price - half, self.price + half

    def z_score(self, reference: float) -> float:
        """(estimate - reference) / standard error. Roughly N(0, 1) for an unbiased estimator
        (Student's t with ``dof`` degrees of freedom when ``dof`` is set)."""
        return (self.price - reference) / self.std_error


@dataclass
class _RunningMoments:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared deviations from the mean

    def update(self, x: np.ndarray) -> None:
        n_b = x.size
        mean_b = float(x.mean())
        m2_b = float(((x - mean_b) ** 2).sum())
        n = self.n + n_b
        delta = mean_b - self.mean
        self.mean += delta * n_b / n
        self.m2 += m2_b + delta**2 * self.n * n_b / n
        self.n = n

    def std_error(self) -> float:
        return float(np.sqrt(self.m2 / (self.n - 1) / self.n))


def payoff(s_t: np.ndarray, k: float, kind: OptionKind) -> np.ndarray:
    if kind == "call":
        return np.maximum(s_t - k, 0.0)
    if kind == "put":
        return np.maximum(k - s_t, 0.0)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def price_european(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "call",
    n_paths: int = 1_000_000,
    seed: Seed = 0,
    chunk: int = DEFAULT_CHUNK,
    *,
    method: Method = "plain",
    **options,
) -> MCResult:
    """Monte Carlo price of a European call or put under GBM.

    ``method`` picks the estimator; ``options`` go to it (see ``mc_engine.variance_reduction``).
    Plain Monte Carlo is the discounted mean payoff with standard error equal to the sample
    standard deviation over sqrt(n_paths), so it shrinks as 1/sqrt(N).
    """
    if method != "plain":
        from mc_engine import variance_reduction as vr

        estimators = {
            "antithetic": vr.antithetic,
            "control_variate": vr.control_variate,
            "sobol": vr.sobol_rqmc,
        }
        if method not in estimators:
            raise ValueError(f"unknown method {method!r}")
        return estimators[method](s0, k, t, r, sigma, q, kind, n_paths, seed, chunk, **options)
    if options:
        raise TypeError(f"plain Monte Carlo takes no options, got {sorted(options)}")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2 to estimate a standard error")
    rng = np.random.default_rng(seed)
    discount = np.exp(-r * t)
    moments = _RunningMoments()
    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        s_t = gbm.terminal(s0, r, sigma, t, rng.standard_normal(m), q=q)
        moments.update(discount * payoff(s_t, k, kind))
        remaining -= m
    return MCResult(price=moments.mean, std_error=moments.std_error(), n_paths=n_paths)
