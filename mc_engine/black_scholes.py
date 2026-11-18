"""Black–Scholes–Merton closed forms with a continuous dividend yield q.

These are the reference values every Monte Carlo estimator in this package is tested against.
All functions broadcast over NumPy arrays.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

OptionKind = Literal["call", "put"]


def _d1_d2(s, k, t, r, sigma, q):
    s, k, t, r, sigma, q = (np.asarray(x, dtype=np.float64) for x in (s, k, t, r, sigma, q))
    vol_sqrt_t = sigma * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    s: ArrayLike,
    k: ArrayLike,
    t: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    kind: OptionKind = "call",
) -> np.ndarray:
    """European option price: C = S e^{-qT} N(d1) - K e^{-rT} N(d2), put by parity."""
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    disc_s = np.asarray(s) * np.exp(-np.asarray(q) * np.asarray(t))
    disc_k = np.asarray(k) * np.exp(-np.asarray(r) * np.asarray(t))
    if kind == "call":
        return disc_s * ndtr(d1) - disc_k * ndtr(d2)
    if kind == "put":
        return disc_k * ndtr(-d2) - disc_s * ndtr(-d1)
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_delta(s, k, t, r, sigma, q=0.0, kind: OptionKind = "call") -> np.ndarray:
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    carry = np.exp(-np.asarray(q) * np.asarray(t))
    return carry * ndtr(d1) if kind == "call" else carry * (ndtr(d1) - 1.0)


def bs_vega(s, k, t, r, sigma, q=0.0) -> np.ndarray:
    """dPrice/dsigma per 1.00 of vol (not per vol point). Same for calls and puts."""
    d1, _ = _d1_d2(s, k, t, r, sigma, q)
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return np.asarray(s) * np.exp(-np.asarray(q) * np.asarray(t)) * pdf * np.sqrt(t)
