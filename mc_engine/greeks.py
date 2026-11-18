"""Monte Carlo delta and vega under GBM: three estimators computed from the same paths.

With S_T = S0 exp((r - q - sigma^2/2) T + sigma sqrt(T) Z) and discounted payoff
Y = e^{-rT} f(S_T):

* Pathwise: differentiate inside the expectation.
      delta = E[e^{-rT} f'(S_T) S_T / S0]
      vega  = E[e^{-rT} f'(S_T) S_T (sqrt(T) Z - sigma T)]
  Low variance, but it needs f to be continuous. For a digital, f' is 0 almost everywhere,
  so pathwise returns 0 with a tiny standard error: confidently wrong.

* Likelihood ratio: differentiate the density of Z instead of the payoff.
      delta = E[Y * Z / (S0 sigma sqrt(T))]
      vega  = E[Y * ((Z^2 - 1) / sigma - Z sqrt(T))]
  Works for any payoff, discontinuous ones included. The weights grow like 1/(sigma sqrt(T)).
  If the payoff doesn't shrink as T -> 0 (in-the-money options, digitals), variance explodes
  at short maturities. At the money the call payoff also shrinks like sqrt(T), so the two
  cancel: the delta variance stays near 5x pathwise (exactly 5 in the T -> 0 limit,
  E[Z^4; Z>0] - 1/4 = 1.25 against 1/4).

* Central bump-and-revalue with common random numbers (CRN): reprice at theta +/- h with the
  same Z. Bias is O(h^2). Reusing Z keeps the variance O(1) as h shrinks for Lipschitz payoffs.
  With independent draws the variance would grow like 1/h^2.

Reference: Glasserman, Monte Carlo Methods in Financial Engineering, ch. 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mc_engine import gbm
from mc_engine.pricer import DEFAULT_CHUNK, Seed, _RunningMoments

GreekKind = Literal["call", "put", "digital_call"]
ESTIMATORS = ("pathwise", "likelihood_ratio", "bump_crn")


@dataclass(frozen=True)
class Estimate:
    value: float
    std_error: float

    def z_score(self, reference: float) -> float:
        return (self.value - reference) / self.std_error


@dataclass(frozen=True)
class GreeksResult:
    delta: dict[str, Estimate]
    vega: dict[str, Estimate]
    n_paths: int


def _payoff(s_t: np.ndarray, k: float, kind: GreekKind) -> np.ndarray:
    if kind == "call":
        return np.maximum(s_t - k, 0.0)
    if kind == "put":
        return np.maximum(k - s_t, 0.0)
    if kind == "digital_call":
        return (s_t > k).astype(np.float64)
    raise ValueError(f"kind must be 'call', 'put' or 'digital_call', got {kind!r}")


def _payoff_derivative(s_t: np.ndarray, k: float, kind: GreekKind) -> np.ndarray:
    """df/dS_T almost everywhere. Zero for the digital: that is why pathwise fails there."""
    if kind == "call":
        return (s_t > k).astype(np.float64)
    if kind == "put":
        return -(s_t < k).astype(np.float64)
    return np.zeros_like(s_t)


def mc_greeks(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: GreekKind = "call",
    n_paths: int = 1 << 18,
    seed: Seed = 0,
    *,
    delta_bump: float | None = None,
    vega_bump: float = 0.01,
    chunk: int = DEFAULT_CHUNK,
) -> GreeksResult:
    """Delta and vega by all three estimators from one set of paths.

    ``delta_bump`` defaults to 1% of spot; ``vega_bump`` is in absolute vol (0.01 = 1 point).
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive for Greeks")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    h_s = 0.01 * s0 if delta_bump is None else delta_bump
    h_v = vega_bump
    if h_s <= 0 or h_v <= 0 or h_v >= sigma:
        raise ValueError("bumps must be positive, and vega_bump smaller than sigma")
    _ = _payoff(np.array([s0]), k, kind)  # validates kind up front

    rng = np.random.default_rng(seed)
    disc = np.exp(-r * t)
    sqrt_t = np.sqrt(t)
    acc = {(g, e): _RunningMoments() for g in ("delta", "vega") for e in ESTIMATORS}

    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        z = rng.standard_normal(m)
        s_t = gbm.terminal(s0, r, sigma, t, z, q=q)
        y = disc * _payoff(s_t, k, kind)
        dfds = disc * _payoff_derivative(s_t, k, kind)

        acc["delta", "pathwise"].update(dfds * s_t / s0)
        acc["vega", "pathwise"].update(dfds * s_t * (sqrt_t * z - sigma * t))

        acc["delta", "likelihood_ratio"].update(y * z / (s0 * sigma * sqrt_t))
        acc["vega", "likelihood_ratio"].update(y * ((z**2 - 1) / sigma - z * sqrt_t))

        # S_T is linear in S0, so the spot bump rescales the same paths.
        up = _payoff(s_t * (s0 + h_s) / s0, k, kind)
        dn = _payoff(s_t * (s0 - h_s) / s0, k, kind)
        acc["delta", "bump_crn"].update(disc * (up - dn) / (2 * h_s))
        up = _payoff(gbm.terminal(s0, r, sigma + h_v, t, z, q=q), k, kind)
        dn = _payoff(gbm.terminal(s0, r, sigma - h_v, t, z, q=q), k, kind)
        acc["vega", "bump_crn"].update(disc * (up - dn) / (2 * h_v))
        remaining -= m

    def pack(greek: str) -> dict[str, Estimate]:
        return {e: Estimate(acc[greek, e].mean, acc[greek, e].std_error()) for e in ESTIMATORS}

    return GreeksResult(delta=pack("delta"), vega=pack("vega"), n_paths=n_paths)
