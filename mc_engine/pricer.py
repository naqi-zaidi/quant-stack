"""Monte Carlo pricing of European options with a reported standard error.

Paths are drawn in fixed-size chunks and combined with Chan's parallel mean/variance update,
so memory stays flat at any path count and the result is identical to a single large draw
from the same generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_engine import gbm
from mc_engine.black_scholes import OptionKind

DEFAULT_CHUNK = 1 << 20


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        half = 1.959963984540054 * self.std_error
        return self.price - half, self.price + half

    def z_score(self, reference: float) -> float:
        """(estimate - reference) / standard error. Roughly N(0, 1) for an unbiased estimator."""
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
    seed: int | np.random.SeedSequence | None = 0,
    chunk: int = DEFAULT_CHUNK,
) -> MCResult:
    """Plain Monte Carlo price of a European call or put under GBM.

    The estimator is the discounted mean payoff; its standard error is the sample standard
    deviation over sqrt(n_paths), so it shrinks as 1/sqrt(N).
    """
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
