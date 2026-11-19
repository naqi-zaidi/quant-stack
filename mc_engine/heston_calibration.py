"""Fast Heston pricing for whole strike grids, and calibration to an implied-vol surface.

Fast pricer: the same Gil-Pelaez inversion as ``heston.heston_price``, but with a
Gauss–Legendre rule on [0, u_max] instead of adaptive quadrature. The characteristic function
is evaluated once per expiry and reused for every strike, so a full surface costs
O(nodes x strikes) per parameter set. Work is in units of the forward (S0 = 1, r = q = 0), so
the phase terms stay O(u ln(K/F)) instead of O(u ln F).

The cutoff and node count adapt to the parameters. With Feller satisfied, |phi(u)| decays
like a Gaussian and u_max ~ 100 suffices. With Feller badly violated, variance piles up near
zero and |phi| decays only exponentially: at T = 2.2 with xi = 1 it is still 5e-2 at u = 81.
So u_max doubles until |phi(u_max)| < 1e-12. The node count then scales with u_max times the
widest |ln K/F|, which sets the oscillation rate: measured across normal and Feller-violating
cases, 0.3 x that product reaches 1e-10 accuracy (in units of the forward), so 0.4 x + 64 is
used. tests/test_heston_calibration.py checks it against the adaptive-quadrature pricer.

Calibration: weighted least squares on vega-scaled price errors,
    residual_i = w_i (model_i - mid_i) / vega_i  ~  w_i (model_iv_i - market_iv_i),
which is the standard first-order proxy for implied-vol errors without inverting model prices
at every iteration. w_i = 1/sqrt(quotes in that expiry), so every expiry carries the same total
weight however densely it is quoted. Several starting points guard against local minima. The
reported fit error is computed exactly at the end, by inverting the fitted model prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from mc_engine.heston import HestonParams, char_func
from mc_engine.implied_vol import implied_vol

PHI_TOL = 1e-12
MIN_NODES, MAX_NODES = 64, 8192
BOUNDS = (np.array([1e-3, 1e-4, 1e-3, -0.999, 1e-4]), np.array([20.0, 1.0, 5.0, 0.999, 1.0]))
DEFAULT_STARTS = (
    (2.0, 0.04, 0.5, -0.7, 0.04),
    (1.0, 0.06, 1.0, -0.8, 0.03),
    (5.0, 0.04, 0.8, -0.6, 0.02),
    (0.5, 0.10, 0.4, -0.5, 0.05),
)


@cache
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def integration_range(t: float, p: HestonParams) -> float:
    """Smallest u_max on a doubling grid with |phi(u_max)| below PHI_TOL."""
    u = 25.0
    while u < 1e5 and abs(char_func(u, 1.0, t, 0.0, p)) > PHI_TOL:
        u *= 2.0
    return u


def call_prices_fwd(k_over_f: np.ndarray, t: float, p: HestonParams) -> np.ndarray:
    """Undiscounted call prices in units of the forward: E[(S_T/F - K/F)^+]."""
    lnk = np.log(np.asarray(k_over_f, dtype=np.float64))
    u_max = integration_range(t, p)
    widest = float(np.abs(lnk).max())
    n = int(np.clip(64 * np.ceil((0.4 * u_max * widest + 64) / 64), MIN_NODES, MAX_NODES))
    x, wq = _gauss_legendre(n)
    u = 0.5 * u_max * (x + 1.0)
    w = 0.5 * u_max * wq
    phi1 = char_func(u - 1j, 1.0, t, 0.0, p)  # phi(-i) = 1 when S0 = F = 1 and r = q = 0
    phi2 = char_func(u, 1.0, t, 0.0, p)
    phase = np.exp(-1j * np.outer(u, lnk))
    iu = (1j * u)[:, None]
    p1 = 0.5 + (w[:, None] * (phase * phi1[:, None] / iu).real).sum(axis=0) / np.pi
    p2 = 0.5 + (w[:, None] * (phase * phi2[:, None] / iu).real).sum(axis=0) / np.pi
    return p1 - np.asarray(k_over_f) * p2


def surface_prices(groups: list[dict], p: HestonParams) -> list[np.ndarray]:
    """Model prices (in currency) for each expiry group of a surface."""
    out = []
    for g in groups:
        call = g["df"] * g["forward"] * call_prices_fwd(g["k_over_f"], g["t"], p)
        put = call - g["df"] * (g["forward"] - g["strike"])  # parity
        out.append(np.where(g["is_put"], put, call))
    return out


def _groups(surface: pd.DataFrame) -> list[dict]:
    groups = []
    for _, g in surface.groupby("t", sort=True):
        vega = g["vega"].to_numpy()
        groups.append(
            {
                "t": float(g["t"].iloc[0]),
                "forward": float(g["forward"].iloc[0]),
                "df": float(g["df"].iloc[0]),
                "strike": g["strike"].to_numpy(),
                "k_over_f": (g["strike"] / g["forward"]).to_numpy(),
                "is_put": (g["kind"] == "put").to_numpy(),
                "mid": g["mid"].to_numpy(),
                # Floor tiny vegas so far-wing quotes can't dominate the fit.
                "vega": np.maximum(vega, 0.01 * vega.max()),
                "weight": 1.0 / np.sqrt(len(g)),
                "index": g.index.to_numpy(),
            }
        )
    return groups


@dataclass(frozen=True)
class CalibrationResult:
    params: HestonParams
    cost: float
    nfev: int
    # (start, fitted params, cost) for every starting point, to show whether they agree.
    runs: tuple[tuple[tuple[float, ...], tuple[float, ...], float], ...] = field(repr=False)
    model_iv: pd.Series = field(repr=False)  # indexed like the input surface

    def iv_errors(self, surface: pd.DataFrame) -> pd.Series:
        """Model minus market implied vol, in vol points (1.0 = 1%)."""
        return 100.0 * (self.model_iv - surface["iv"])


def model_implied_vols(surface: pd.DataFrame, p: HestonParams) -> pd.Series:
    groups = _groups(surface)
    vols = pd.Series(np.nan, index=surface.index)
    for g, prices in zip(groups, surface_prices(groups, p), strict=True):
        r = -np.log(g["df"]) / g["t"]
        for i, px, k, put in zip(g["index"], prices, g["strike"], g["is_put"], strict=True):
            vols[i] = implied_vol(px, g["forward"], k, g["t"], r, r, "put" if put else "call")
    return vols


def calibrate(
    surface: pd.DataFrame, starts: tuple[tuple[float, ...], ...] = DEFAULT_STARTS
) -> CalibrationResult:
    """Fit (kappa, theta, xi, rho, v0) to a surface from ``marketdata.option_chain``."""
    groups = _groups(surface)

    def residuals(x: np.ndarray) -> np.ndarray:
        p = HestonParams(*x)
        return np.concatenate(
            [
                g["weight"] * (model - g["mid"]) / g["vega"]
                for g, model in zip(groups, surface_prices(groups, p), strict=True)
            ]
        )

    best, nfev, runs = None, 0, []
    for x0 in starts:
        fit = least_squares(
            residuals,
            np.asarray(x0, dtype=float),
            bounds=BOUNDS,
            method="trf",
            x_scale="jac",
            max_nfev=400,
        )
        nfev += fit.nfev
        runs.append((tuple(x0), tuple(float(v) for v in fit.x), float(fit.cost)))
        if best is None or fit.cost < best.cost:
            best = fit
    params = HestonParams(*(float(v) for v in best.x))
    return CalibrationResult(
        params=params,
        cost=float(best.cost),
        nfev=nfev,
        runs=tuple(runs),
        model_iv=model_implied_vols(surface, params),
    )
