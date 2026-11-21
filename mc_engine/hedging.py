"""Discrete delta hedging of a short European option: P&L, its gamma decomposition, and costs.

A dealer sells one option at implied vol sigma_i, takes the Black–Scholes premium, and holds the
Black–Scholes delta at a hedging vol sigma_h, rebalanced at n evenly spaced dates. Cash earns r.
At expiry the option is physically settled, so the only closing trade is the gap between the last
hedge and the share delivered. Every amount is reported in time-0 money.

What the simulator lets you check:

* Martingale identity. When the stock drifts at r, the discounted stock is a martingale, so the
  hedge's discounted gains have mean zero whatever the hedge ratio and however often it changes:
  E[P&L] = BS(sigma_i) - BS(sigma_r) exactly, for every n and every hedging vol (before costs).
* Gamma P&L. Marking the option at the hedging vol, each period earns, to second order,
  ½ Γ_h S² (sigma_h² dt − R²), R being the period's discounted return. Summed, and with R²
  replaced by its mean sigma_r² dt, that is (sigma_h² − sigma_r²) · Σ ½ Γ_h S² dt: the variance
  spread times the path's dollar-gamma exposure. So as n grows,
      P&L → BS(sigma_i) − BS(sigma_h) + (sigma_h² − sigma_r²) · dollar_gamma.
  Hedge at implied vol and the first term vanishes: P&L depends on the path's dollar gamma.
  Hedge at realized vol and the second term vanishes: P&L tends to the fixed amount
  BS(sigma_i) − BS(sigma_r).
* Discretisation error. With sigma_i = sigma_r = sigma_h, the P&L s.d. falls like 1/√n:
  Derman & Kamal's sqrt(π / (4n)) · vega · sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_engine.black_scholes import OptionKind, bs_delta, bs_gamma, bs_price, bs_vega


@dataclass(frozen=True)
class HedgeRun:
    """One hedging policy applied to every path. Arrays have one entry per path."""

    pnl: np.ndarray  # terminal P&L before costs, time-0 money
    gamma_pnl: np.ndarray  # Σ ½ Γ_h S² (sigma_h² dt − R²), time-0 money
    dollar_gamma: np.ndarray  # Σ ½ Γ_h S² dt, time-0 money
    traded: np.ndarray  # Σ |shares traded| · price, opening and closing trades included
    opening: float  # |Δ_0| · S_0, the opening trade (the same on every path)
    locked_in: float  # BS(sigma_i) − BS(sigma_h): the premium gap fixed at inception
    n_rebalances: int

    def net(self, cost: float) -> np.ndarray:
        """P&L after paying ``cost`` (a fraction of notional, 5 bp = 5e-4) on every trade."""
        return self.pnl - cost * self.traded

    def costs(self, cost: float) -> np.ndarray:
        return cost * self.traded

    @property
    def rebalance_traded(self) -> np.ndarray:
        """Turnover after the opening trade: what rebalancing more often costs."""
        return self.traded - self.opening

    @property
    def residual(self) -> np.ndarray:
        """What the second-order decomposition misses: third-order and timing terms."""
        return self.pnl - self.locked_in - self.gamma_pnl

    def mean_se(self, x: np.ndarray | None = None) -> tuple[float, float]:
        x = self.pnl if x is None else x
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def delta_hedge(
    paths: np.ndarray,
    k: float,
    t: float,
    r: float,
    sigma_implied: float,
    sigma_hedge: float,
    n_rebalances: int,
    kind: OptionKind = "call",
) -> HedgeRun:
    """Sell one option at ``sigma_implied`` and delta-hedge it at ``sigma_hedge``.

    ``paths`` has shape (n_paths, n_steps + 1) on an even grid over [0, t]; the hedge is
    rebalanced at every (n_steps / n_rebalances)-th point, so one set of paths serves every n.
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1)")
    if np.any(paths <= 0):
        raise ValueError("prices must be positive")
    n_steps = paths.shape[1] - 1
    if n_rebalances < 1 or n_steps % n_rebalances:
        raise ValueError(f"n_rebalances must divide the {n_steps} path steps")
    if sigma_implied <= 0 or sigma_hedge <= 0 or t <= 0:
        raise ValueError("vols and maturity must be positive")

    s = paths[:, :: n_steps // n_rebalances]
    dt = t / n_rebalances
    growth = np.exp(r * dt)
    m = len(s)

    premium = float(bs_price(s[0, 0], k, t, r, sigma_implied, kind=kind))
    cash = np.full(m, premium)
    pos = np.zeros(m)
    traded = np.zeros(m)
    gamma_pnl = np.zeros(m)
    dollar_gamma = np.zeros(m)
    opening = 0.0
    for j in range(n_rebalances):
        tau, disc = t - j * dt, np.exp(-r * j * dt)
        delta = bs_delta(s[:, j], k, tau, r, sigma_hedge, kind=kind)
        trade = delta - pos
        if j == 0:
            opening = float(np.abs(trade[0]) * s[0, 0])
        cash = (cash - trade * s[:, j]) * growth
        traded += disc * np.abs(trade) * s[:, j]
        pos = delta

        half_gamma_s2 = 0.5 * bs_gamma(s[:, j], k, tau, r, sigma_hedge) * s[:, j] ** 2
        ret = s[:, j + 1] / (s[:, j] * growth) - 1.0
        gamma_pnl += disc * half_gamma_s2 * (sigma_hedge**2 * dt - ret**2)
        dollar_gamma += disc * half_gamma_s2 * dt

    s_t = s[:, -1]
    if kind == "call":
        payoff, delivered = np.maximum(s_t - k, 0.0), (s_t > k).astype(float)
    else:
        payoff, delivered = np.maximum(k - s_t, 0.0), -(s_t < k).astype(float)
    disc_t = np.exp(-r * t)
    traded += disc_t * np.abs(pos - delivered) * s_t
    return HedgeRun(
        pnl=disc_t * (cash + pos * s_t - payoff),
        gamma_pnl=gamma_pnl,
        dollar_gamma=dollar_gamma,
        traded=traded,
        opening=opening,
        locked_in=premium - float(bs_price(s[0, 0], k, t, r, sigma_hedge, kind=kind)),
        n_rebalances=n_rebalances,
    )


def derman_kamal_sd(s0: float, k: float, t: float, r: float, sigma: float, n: int) -> float:
    """Hedging-error s.d. for n rebalances when every vol equals sigma: sqrt(π/(4n))·vega·sigma.

    Derman & Kamal (1999), "When You Cannot Hedge Continuously"; the same constant is used by
    QuantLib's DiscreteHedging example. Asymptotic in n.
    """
    return float(np.sqrt(np.pi / (4 * n)) * bs_vega(s0, k, t, r, sigma) * sigma)


def gbm_grid(
    s0: float, mu: float, sigma: float, t: float, n_steps: int, n_paths: int, seed: int
) -> np.ndarray:
    """Exact GBM paths at drift ``mu``. One seed gives the same normals for every sigma,
    so paths at different vols are common-random-number pairs."""
    z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    dt = t / n_steps
    log_s = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    return s0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), log_s], axis=1))
