"""Fitting Avellaneda–Stoikov's inputs to the market (Project 3, step 3).

Avellaneda and Stoikov need two things from the market:

- sigma, the volatility of the mid. It is estimated from mid changes over `step_s`-second
  intervals: the realised variance per second in each session, averaged over sessions (ticks
  per sqrt-second). Short intervals pick up the bid-ask bounce, which inflates the estimate;
  `signature()` shows how much, interval by interval.
- How fast fills thin out with distance from the mid, lambda(delta) = A exp(-k delta). This is
  the rate at which a resting quote delta ticks from the mid would be reached. For every
  aggressive order, the estimate takes the furthest it traded from the mid that stood just
  before it, then counts the orders reaching at least delta, per side and per second. A
  least-squares line through log lambda against delta gives k (its slope) and A. Both are
  averaged over sessions, with standard errors across them.

Calibration uses its own sessions (seeds 1001 on, without a maker), separate from the sessions
the strategies are evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_strategy.market import NS, TRADING_YEAR_HOURS, MarketParams, Session


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float  # across sessions
    per_session: tuple[float, ...]


def sigma_per_second(session: Session, step_s: float) -> float:
    """Realised volatility of the mid over `step_s`-second intervals, in ticks per sqrt(s)."""
    p = session.params
    n = int(p.hours * 3600 / step_s)
    grid = (np.arange(n + 1) * step_s * NS).astype(np.int64)
    grid[0] = 1  # the first quote arrives just after the open
    mid = session.mid_at(grid) / p.tick
    d = np.diff(mid)
    d = d[np.isfinite(d)]
    return float(np.sqrt((d**2).sum() / (len(d) * step_s)))


def estimate_sigma(sessions: list[Session], step_s: float = 60.0) -> Estimate:
    xs = [sigma_per_second(s, step_s) for s in sessions]
    return Estimate(float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(len(xs))), tuple(xs))


def signature(sessions: list[Session], steps_s: tuple[float, ...]) -> dict[float, Estimate]:
    """Volatility estimate at each sampling interval (the 'signature plot')."""
    return {h: estimate_sigma(sessions, h) for h in steps_s}


def true_sigma(p: MarketParams) -> float:
    """The fair value's volatility in ticks per sqrt(s) at the opening price."""
    per_second = p.sigma / np.sqrt(TRADING_YEAR_HOURS * 3600)
    return float(per_second * p.s0 * 10_000 / p.tick)


def reach_per_order(session: Session) -> np.ndarray:
    """For every aggressive order, how far from the prior mid it traded, in ticks."""
    tr = session.trades
    if len(tr["ts"]) == 0:
        return np.array([])
    taker = tr["taker"]
    # Trades of one order are consecutive and share its id.
    starts = np.flatnonzero(np.r_[True, taker[1:] != taker[:-1]])
    ts = tr["ts"][starts].astype(np.int64)
    mid = session.mid_at(ts - 1) / session.params.tick
    side = tr["taker_side"][starts].astype(float)
    px = tr["price"].astype(float) / session.params.tick
    worst = np.where(side > 0, np.maximum.reduceat(px, starts), np.minimum.reduceat(px, starts))
    reach = side * (worst - mid)
    return reach[np.isfinite(reach)]


def intensity_curve(session: Session, deltas: np.ndarray) -> np.ndarray:
    """lambda(delta): aggressive orders per second, per side, reaching at least delta ticks."""
    reach = reach_per_order(session)
    seconds = session.params.hours * 3600
    return np.array([(reach >= d).sum() for d in deltas]) / seconds / 2


@dataclass(frozen=True)
class IntensityFit:
    k: Estimate  # per tick
    A: Estimate  # per second, per side
    r2: float  # of the log-linear fit to the pooled curve
    deltas: np.ndarray
    curve: np.ndarray  # pooled lambda(delta)


def estimate_intensity(sessions: list[Session],
                       deltas: np.ndarray | None = None) -> IntensityFit:
    if deltas is None:
        deltas = np.arange(0.5, 6.01, 0.5)
    ks, As, curves = [], [], []
    for s in sessions:
        lam = intensity_curve(s, deltas)
        ok = lam > 0
        slope, icept = np.polyfit(deltas[ok], np.log(lam[ok]), 1)
        ks.append(-slope)
        As.append(float(np.exp(icept)))
        curves.append(lam)
    pooled = np.mean(curves, axis=0)
    ok = pooled > 0
    slope, icept = np.polyfit(deltas[ok], np.log(pooled[ok]), 1)
    fit = icept + slope * deltas[ok]
    y = np.log(pooled[ok])
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def est(xs):
        return Estimate(float(np.mean(xs)), float(np.std(xs, ddof=1) / np.sqrt(len(xs))),
                        tuple(float(x) for x in xs))

    return IntensityFit(est(ks), est(As), float(r2), deltas, pooled)
