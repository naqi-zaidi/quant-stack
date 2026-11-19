"""Black–Scholes implied volatility by bracketed root-finding.

The call price is strictly increasing in sigma for T > 0, so the root is unique whenever the
price sits strictly inside the no-arbitrage bounds. Brent's method on a wide bracket always
converges. Newton would be faster, but it can overshoot far from the money where vega is tiny.
Prices outside the bounds return NaN rather than a made-up volatility.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from mc_engine.black_scholes import OptionKind, bs_price

VOL_LO, VOL_HI = 1e-6, 5.0


def implied_vol(
    price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float = 0.0,
    kind: OptionKind = "call",
    tol: float = 1e-12,
) -> float:
    disc_s, disc_k = s * np.exp(-q * t), k * np.exp(-r * t)
    if kind == "call":
        lower, upper = max(disc_s - disc_k, 0.0), disc_s
    elif kind == "put":
        lower, upper = max(disc_k - disc_s, 0.0), disc_k
    else:
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not lower < price < upper:
        return float("nan")

    def gap(sigma: float) -> float:
        return float(bs_price(s, k, t, r, sigma, q, kind)) - price

    if gap(VOL_LO) > 0 or gap(VOL_HI) < 0:
        return float("nan")  # inside the bounds but beyond the bracket: numerically degenerate
    return float(brentq(gap, VOL_LO, VOL_HI, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=200))
