"""Heston (1993) stochastic volatility: semi-analytic prices and two simulation schemes.

    dS / S = (r - q) dt + sqrt(v) dW_S
    dv     = kappa (theta - v) dt + xi sqrt(v) dW_v,      d<W_S, W_v> = rho dt

Semi-analytic price. The characteristic function of ln S_T is known in closed form. This uses
the "little Heston trap" form of Albrecher et al. (2007), which avoids the branch-cut jumps of
Heston's original form at long maturities. Prices follow by Gil-Pelaez inversion:
    C = S0 e^{-qT} P1 - K e^{-rT} P2,
    P2 = 1/2 + 1/pi int_0^inf Re[e^{-iu ln K} phi(u) / (iu)] du,
    P1 = 1/2 + 1/pi int_0^inf Re[e^{-iu ln K} phi(u - i) / (iu phi(-i))] du.

Simulation is not exact. v can hit zero, and when the Feller condition 2 kappa theta >= xi^2
fails it does so often. Two schemes:

* Full-truncation Euler (Lord, Koekkoek and van Dijk, 2010): use max(v, 0) wherever v appears.
  Simple, but the time-discretisation bias is large when Feller fails.
* QE (Andersen, 2008): match the first two moments of v_{t+dt} given v_t, with a quadratic of
  a normal when the variance is well away from zero, or a point mass at 0 plus an exponential
  near zero. The log-price step integrates the variance with a trapezoid. This is far more
  accurate at coarse steps. It is not exactly a martingale, since Andersen's martingale
  correction is not applied, which leaves a small residual drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from mc_engine.black_scholes import OptionKind
from mc_engine.pricer import MCResult, Seed, _RunningMoments, payoff

Scheme = Literal["euler", "qe"]
PSI_CRIT = 1.5  # Andersen's switching threshold between the two QE branches


@dataclass(frozen=True)
class HestonParams:
    kappa: float  # mean-reversion speed of variance
    theta: float  # long-run variance
    xi: float  # volatility of variance ("vol of vol")
    rho: float  # spot / variance correlation
    v0: float  # initial variance

    @property
    def feller(self) -> bool:
        """True when 2 kappa theta >= xi^2, i.e. v stays strictly positive."""
        return 2 * self.kappa * self.theta >= self.xi**2


def char_func(u, s0: float, t: float, r: float, p: HestonParams, q: float = 0.0):
    """E[exp(i u ln S_T)], little-trap formulation. Vectorised over complex ``u``."""
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b**2 + p.xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * t)
    c = (r - q) * iu * t + (p.kappa * p.theta / p.xi**2) * (
        (b - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
    )
    dd = (b - d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * np.log(s0) + c + dd * p.v0)


def heston_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    p: HestonParams,
    q: float = 0.0,
    kind: OptionKind = "call",
) -> float:
    """Semi-analytic price by Gil-Pelaez inversion (adaptive quadrature, ~1e-10 accuracy)."""
    lnk = np.log(k)
    fwd = char_func(-1j, s0, t, r, p, q).real  # = S0 e^{(r-q)T}

    def p1_integrand(u: float) -> float:
        return float(
            (np.exp(-1j * u * lnk) * char_func(u - 1j, s0, t, r, p, q) / (1j * u * fwd)).real
        )

    def p2_integrand(u: float) -> float:
        return float((np.exp(-1j * u * lnk) * char_func(u, s0, t, r, p, q) / (1j * u)).real)

    opts = {"limit": 500, "epsabs": 1e-10, "epsrel": 1e-10}
    p1 = 0.5 + quad(p1_integrand, 0.0, np.inf, **opts)[0] / np.pi
    p2 = 0.5 + quad(p2_integrand, 0.0, np.inf, **opts)[0] / np.pi
    call = s0 * np.exp(-q * t) * p1 - k * np.exp(-r * t) * p2
    if kind == "call":
        return float(call)
    if kind == "put":
        return float(call - s0 * np.exp(-q * t) + k * np.exp(-r * t))
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def effective_vol(t: float, p: HestonParams) -> float:
    """sqrt of the average expected variance over [0, T]: the xi -> 0 Black–Scholes vol."""
    integrated = p.theta * t + (p.v0 - p.theta) * (1.0 - np.exp(-p.kappa * t)) / p.kappa
    return float(np.sqrt(integrated / t))


def simulate_log_terminal(
    s0: float,
    t: float,
    r: float,
    p: HestonParams,
    q: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    scheme: Scheme = "qe",
) -> np.ndarray:
    """ln S_T for ``n_paths`` paths. Memory is O(n_paths): steps are applied in place."""
    if scheme not in ("euler", "qe"):
        raise ValueError(f"scheme must be 'euler' or 'qe', got {scheme!r}")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    dt = t / n_steps
    x = np.full(n_paths, np.log(s0))
    v = np.full(n_paths, p.v0)
    rho_bar = np.sqrt(1.0 - p.rho**2)

    if scheme == "euler":
        for _ in range(n_steps):
            z_v = rng.standard_normal(n_paths)
            z_x = p.rho * z_v + rho_bar * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)
            sq = np.sqrt(vp * dt)
            x += (r - q - 0.5 * vp) * dt + sq * z_x
            v += p.kappa * (p.theta - vp) * dt + p.xi * sq * z_v
        return x

    ek = np.exp(-p.kappa * dt)
    g1 = g2 = 0.5
    k0 = -p.rho * p.kappa * p.theta * dt / p.xi
    k1 = g1 * dt * (p.kappa * p.rho / p.xi - 0.5) - p.rho / p.xi
    k2 = g2 * dt * (p.kappa * p.rho / p.xi - 0.5) + p.rho / p.xi
    k3 = g1 * dt * (1.0 - p.rho**2)
    k4 = g2 * dt * (1.0 - p.rho**2)
    for _ in range(n_steps):
        z_v = rng.standard_normal(n_paths)
        z_x = rng.standard_normal(n_paths)
        m = p.theta + (v - p.theta) * ek
        s2 = v * p.xi**2 * ek * (1.0 - ek) / p.kappa + p.theta * p.xi**2 * (1.0 - ek) ** 2 / (
            2.0 * p.kappa
        )
        psi = s2 / m**2
        v_new = np.empty_like(v)

        quad_branch = psi <= PSI_CRIT
        pq = psi[quad_branch]
        inv = 2.0 / pq
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
        a = m[quad_branch] / (1.0 + b2)
        v_new[quad_branch] = a * (np.sqrt(b2) + z_v[quad_branch]) ** 2

        exp_branch = ~quad_branch
        pe = psi[exp_branch]
        prob0 = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - prob0) / m[exp_branch]
        u = ndtr(z_v[exp_branch])
        one_minus_u = ndtr(-z_v[exp_branch])  # 1 - U without cancellation
        v_new[exp_branch] = np.where(
            u <= prob0, 0.0, np.log((1.0 - prob0) / np.maximum(one_minus_u, 1e-300)) / beta
        )

        x += (r - q) * dt + k0 + k1 * v + k2 * v_new + np.sqrt(k3 * v + k4 * v_new) * z_x
        v = v_new
    return x


def price_heston_mc(
    s0: float,
    k: float,
    t: float,
    r: float,
    p: HestonParams,
    q: float = 0.0,
    kind: OptionKind = "call",
    n_steps: int = 64,
    n_paths: int = 1 << 17,
    seed: Seed = 0,
    *,
    scheme: Scheme = "qe",
    chunk: int = 1 << 16,
) -> MCResult:
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * t)
    moments = _RunningMoments()
    left = n_paths
    while left > 0:
        m = min(chunk, left)
        s_t = np.exp(simulate_log_terminal(s0, t, r, p, q, n_steps, m, rng, scheme))
        moments.update(disc * payoff(s_t, k, kind))
        left -= m
    return MCResult(moments.mean, moments.std_error(), n_paths, method=f"heston_{scheme}")
