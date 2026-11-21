"""Order-flow signals for the market maker, fitted walk-forward (Project 3, step 4).

At each decision the maker sees everyone else's book (maker.MarketView). From the sequence of
those views, FlowFeatures computes three numbers, using only what was visible at the time:

  ofi              Cont–Kukanov–Stoikov order-flow imbalance (arXiv 1011.6402), summed over
                   the last `ofi_window_s` seconds. It is computed from consecutive decision
                   snapshots of the best quotes, in hundreds of shares.
  depth_imbalance  (D_bid - D_ask) / (D_bid + D_ask), where D is the shares resting within
                   `depth_ticks` ticks of that side's best price (best level included).
  last_move        Change in the mid over the last `lag_s` seconds, in ticks.

The maker's own orders never enter them: MarketView has already taken them out.

FlowModel predicts the change in the mid over the next `horizon_s` seconds as a linear
combination of the three. It is fitted by least squares with no intercept, because the market
has no drift, so an intercept would only fit noise as a constant lean.

Forecasts are scored against the forecast of no change, the quote a maker without a signal
implicitly makes: R²_oos = 1 - sum (y - ŷ)² / sum y². Zero means no better than no change;
negative means worse.

walk_forward() fits and trades in time order. Each day is traded with a model fitted only on
the history block and earlier days, and then its log joins the training set.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from mm_strategy.maker import AvellanedaStoikov, MarketView, Quotes

FEATURES = ("ofi", "depth_imbalance", "last_move")


@dataclass
class FlowFeatures:
    """Online features from successive MarketViews. Call update() once per decision, in order;
    decisions must come every `step_s` seconds. Every call is logged (`log`) so the session can
    later be used as training data, with exactly the numbers the strategy saw."""

    step_s: float = 0.1
    ofi_window_s: float = 5.0
    depth_ticks: int = 3
    lag_s: float = 1.0
    log: list[tuple[int, float, float, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._w = round(self.ofi_window_s / self.step_s)
        self._lag = round(self.lag_s / self.step_s)
        self._e: deque[float] = deque(maxlen=self._w)
        self._mids: deque[float] = deque(maxlen=self._lag + 1)
        self._prev: tuple[int, int, int, int] | None = None

    @staticmethod
    def _depth(levels: tuple[tuple[int, int], ...], ticks: int, tick: int) -> int:
        best = levels[0][0]
        return sum(q for px, q in levels if abs(px - best) < ticks * tick)

    def update(self, v: MarketView) -> np.ndarray | None:
        """The features at this decision, or None while the windows fill (the first
        `ofi_window_s` seconds) or when either side of the book is empty."""
        two = bool(v.bids) and bool(v.asks)
        snap = (v.bids[0][0], v.bids[0][1], v.asks[0][0], v.asks[0][1]) if two else None
        e = 0.0
        if snap is not None and self._prev is not None:
            b, bq, a, aq = snap
            pb, pbq, pa, paq = self._prev
            e = ((bq if b >= pb else 0) - (pbq if b <= pb else 0)
                 - (aq if a <= pa else 0) + (paq if a >= pa else 0))
        self._prev = snap
        self._e.append(e)
        mid = (snap[0] + snap[2]) / (2 * v.tick) if snap is not None else math.nan
        self._mids.append(mid)

        x = None
        if (snap is not None and len(self._e) == self._w and len(self._mids) == self._lag + 1
                and math.isfinite(self._mids[0])):
            db = self._depth(v.bids, self.depth_ticks, v.tick)
            da = self._depth(v.asks, self.depth_ticks, v.tick)
            x = np.array([sum(self._e) / 100, (db - da) / (db + da), mid - self._mids[0]])
        nan = math.nan
        self.log.append((v.now, mid, *(x if x is not None else (nan, nan, nan))))
        return x


ALL = tuple(range(len(FEATURES)))


def training_rows(log, horizon_s: float, step_s: float = 0.1,
                  columns: tuple[int, ...] = ALL) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) from one session's log: the features (those in `columns`) at each decision, and
    the mid change over the next `horizon_s` seconds. Rows without both are dropped."""
    a = np.asarray(log, dtype=float).reshape(-1, 2 + len(FEATURES))
    h = round(horizon_s / step_s)
    if len(a) <= h:
        return np.empty((0, len(columns))), np.empty(0)
    x, mid = a[:-h, [2 + c for c in columns]], a[:, 1]
    y = mid[h:] - mid[:-h]
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[ok], y[ok]


@dataclass(frozen=True)
class FlowModel:
    """Predicted mid change over `horizon_s` = coef · features, in ticks."""

    coef: tuple[float, ...]
    horizon_s: float
    rows: int  # training rows
    sessions: int  # training sessions
    columns: tuple[int, ...] = ALL  # which features (indices into FEATURES) it uses

    def predict(self, x: np.ndarray) -> float:
        """x: all the features, as FlowFeatures.update returns them."""
        return float(np.dot(self.coef, x[list(self.columns)]))


def fit(logs: list, horizon_s: float, step_s: float = 0.1,
        columns: tuple[int, ...] = ALL) -> FlowModel:
    """Least squares, no intercept, pooled over sessions."""
    parts = [training_rows(log, horizon_s, step_s, columns) for log in logs]
    x = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return FlowModel(tuple(float(c) for c in coef), horizon_s, len(y), len(logs), columns)


def r2_oos(model: FlowModel, log, step_s: float = 0.1) -> float:
    """1 - SSE / sum y²: the forecast against no change, on a session the model never saw."""
    x, y = training_rows(log, model.horizon_s, step_s, model.columns)
    resid = y - x @ np.asarray(model.coef)
    return float(1 - (resid**2).sum() / (y**2).sum())


def leave_one_out_r2(logs: list, horizon_s: float, step_s: float = 0.1,
                     columns: tuple[int, ...] = ALL) -> np.ndarray:
    """R²_oos of each session under a model fitted on all the others."""
    return np.array([r2_oos(fit(logs[:i] + logs[i + 1:], horizon_s, step_s, columns), logs[i],
                            step_s) for i in range(len(logs))])


@dataclass
class FlowSkewAS:
    """Avellaneda–Stoikov whose reservation price is shifted by the flow model's predicted mid
    change: r = s + weight * ŷ - q γ σ² τ. With model=None it quotes exactly as plain A–S and
    only records features. The strategy holds state, so build a fresh one for every session."""

    base: AvellanedaStoikov
    model: FlowModel | None = None
    weight: float = 1.0
    features: FlowFeatures = field(default_factory=FlowFeatures)
    shifts: list[float] = field(default_factory=list)  # the shift applied at each decision

    def quote(self, v: MarketView) -> Quotes:
        x = self.features.update(v)
        shift = 0.0
        if self.model is not None and x is not None:
            shift = self.weight * self.model.predict(x)
        self.shifts.append(shift)
        return self.base.quote_about(v, shift)


def walk_forward(history: list, days, trade, horizon_s: float, step_s: float = 0.1) -> list:
    """Expanding-window walk-forward. For each day in order: fit on every log before it (the
    history block, then the earlier days), call trade(day, model) -> (result, log), and add
    that log to the training set. Returns [(day, model, result)]. No day's model can see the
    day itself or anything after it."""
    logs = list(history)
    out = []
    for day in days:
        model = fit(logs, horizon_s, step_s)
        result, log = trade(day, model)
        out.append((day, model, result))
        logs.append(log)
    return out
