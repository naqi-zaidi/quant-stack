"""A risk overlay for the market maker, ported from my final-year project (Project 3, step 6).

The FYP (github.com/naqi-zaidi/fyp-trading-risk-guardrails, v10) is a real-time risk
controller for a directional EUR/USD strategy in MetaTrader 5. On every one-minute bar it
labels the volatility regime, scores anomalies, and a finite-state machine decides whether the
strategy may trade at full size (Allow), reduced size (Throttle), not at all (Pause), or with a
small hedge (Hedge). Its parameters are frozen in its config.yaml.

Ported here, on bars built from the maker's own 100 ms view of the mid, `bar_s` seconds long
(5 s instead of the FYP's one minute; everything counted in bars keeps the FYP's value):
- features, all incremental so each bar uses only data up to it: the bar's mid change and
  true range in ticks, Wilder ATR over 14 and 28 bars, range ratio = TR / ATR28, rolling
  standard deviations of the change over 20 and 60 bars, the MACD(12, 26, 9) histogram as the
  slope, and rolling z-scores (window 30) of the change and the range ratio;
- the regime detector: k-means (k = 5, 10 restarts) on [vol_20, range_ratio, slope], fitted
  on the history block, with clusters named by the FYP's ranking (lowest volatility calm; the
  next two trend up / trend down by slope; the top two volatile / stressed by range ratio).
  Entering volatile or stressed needs the CUSUM gate (k = 1.5 σ60, h = 3 σ60) to fire, and a
  new label must persist 2 bars;
- anomaly levels: moderate when either z-score reaches 2, severe at 3, persisting 1 bar;
- the policy state machine with the FYP's rule precedence: Pause on a stressed regime with a
  severe anomaly; escalation from Throttle to Pause; Throttle on a volatile, stressed or
  trending regime with a moderate or severe anomaly; back to Allow when a timer runs out or
  after 3 clear bars. Throttle lasts 5 bars at 0.33x size, Pause 10 bars, and leaving either
  starts a cooldown of the same length.
Left out: the Isolation Forest (advisory in the FYP; it needs scikit-learn), the economic
calendar (there is no scheduled news here), and the Hedge action (it opens a directional
position, which a post-only maker can't).

For a maker, Allow quotes as the strategy says, Throttle cuts both sizes to a third, and Pause
quotes only the side that reduces inventory. Two limits sit on top, as the plan for this step
asked:
- an inventory cap: at or beyond it, only the reducing side is quoted;
- a drawdown kill switch: once the session's PnL, marked to the mid, falls `max_drawdown`
  below its running peak, the maker only reduces inventory for the rest of the session.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from mm_strategy.maker import MarketView, Quotes

# ---------------------------------------------------------------- rolling statistics (FYP core)


class RollingStats:
    """Mean and standard deviation over the last `window` values, from running sums (the FYP's
    WelfordRolling). The z-score of a value is taken after it joins the window, as there."""

    def __init__(self, window: int) -> None:
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sq = 0.0

    def update(self, x: float) -> None:
        if len(self._buf) == self.window:
            old = self._buf[0]
            self._sum -= old
            self._sq -= old * old
        self._buf.append(x)
        self._sum += x
        self._sq += x * x

    @property
    def mean(self) -> float:
        return self._sum / len(self._buf) if self._buf else 0.0

    @property
    def std(self) -> float:
        n = len(self._buf)
        if n < 2:
            return 0.0
        mu = self._sum / n
        return math.sqrt(max(0.0, self._sq / n - mu * mu))

    def z(self, x: float) -> float:
        s = self.std
        return 0.0 if s < 1e-15 else (x - self.mean) / s


class WilderATR:
    """ATR_t = ((n - 1) ATR_{t-1} + TR_t) / n, started from the mean of the first n."""

    def __init__(self, window: int) -> None:
        self.window = window
        self.value = 0.0
        self._warm: list[float] = []
        self.ready = False

    def update(self, tr: float) -> None:
        if not self.ready:
            self._warm.append(tr)
            if len(self._warm) >= self.window:
                self.value = sum(self._warm) / self.window
                self.ready = True
            return
        self.value = ((self.window - 1) * self.value + tr) / self.window


class EMA:
    def __init__(self, span: int) -> None:
        self.alpha = 2.0 / (span + 1)
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class MACD:
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast, self.slow, self.signal = EMA(fast), EMA(slow), EMA(signal)
        self.histogram = 0.0

    def update(self, x: float) -> float:
        line = self.fast.update(x) - self.slow.update(x)
        self.histogram = line - self.signal.update(line)
        return self.histogram


class CUSUM:
    """Two one-sided CUSUMs: S+ = max(0, S+ + x - k), S- = max(0, S- - x - k), with k and h
    scaled by a rolling standard deviation; it fires while either exceeds h."""

    def __init__(self, k_mult: float = 1.5, h_mult: float = 3.0) -> None:
        self.k_mult, self.h_mult = k_mult, h_mult
        self.reset()

    def reset(self) -> None:
        self.s_plus = self.s_minus = 0.0

    def update(self, x: float, sigma: float) -> bool:
        k, h = self.k_mult * sigma, self.h_mult * sigma
        self.s_plus = max(0.0, self.s_plus + x - k)
        self.s_minus = max(0.0, self.s_minus - x - k)
        return self.s_plus > h or self.s_minus > h


# ---------------------------------------------------------------- bars and features


@dataclass(frozen=True)
class BarFeatures:
    ret: float  # mid change over the bar, ticks
    tr: float  # true range, ticks
    range_ratio: float  # tr / ATR28 (0 until the ATR is ready)
    vol_20: float
    sigma_60: float
    slope: float  # MACD histogram
    z_ret: float
    z_range: float
    ready: bool  # every window full (60 bars)

    def regime_vector(self) -> np.ndarray:
        return np.array([self.vol_20, self.range_ratio, self.slope])


class FeatureEngine:
    """The FYP's FeatureEngineer, on bars of the mid in ticks."""

    def __init__(self) -> None:
        self.r20, self.r60 = RollingStats(20), RollingStats(60)
        self.atr14, self.atr28 = WilderATR(14), WilderATR(28)
        self.macd = MACD()
        self.z_ret, self.z_range = RollingStats(30), RollingStats(30)
        self.prev_close: float | None = None
        self.bars = 0

    def update(self, o: float, h: float, lo: float, c: float) -> BarFeatures:
        self.bars += 1
        ret = 0.0 if self.prev_close is None else c - self.prev_close
        tr = h - lo if self.prev_close is None else max(h - lo, abs(h - self.prev_close),
                                                         abs(lo - self.prev_close))
        self.prev_close = c
        self.r20.update(ret)
        self.r60.update(ret)
        self.atr14.update(tr)
        self.atr28.update(tr)
        rr = tr / self.atr28.value if self.atr28.value > 0 else 0.0
        slope = self.macd.update(c)
        self.z_ret.update(ret)
        self.z_range.update(rr)
        return BarFeatures(ret, tr, rr, self.r20.std, self.r60.std, slope, self.z_ret.z(ret),
                           self.z_range.z(rr), self.bars >= 60)


class BarBuilder:
    """Turns mids sampled every decision into OHLC bars of `per_bar` decisions."""

    def __init__(self, per_bar: int) -> None:
        self.per_bar = per_bar
        self.n = 0
        self.o = self.h = self.lo = self.c = math.nan

    def add(self, mid: float | None) -> tuple[float, float, float, float] | None:
        self.n += 1
        if mid is not None and math.isfinite(mid):
            if math.isnan(self.o):
                self.o = self.h = self.lo = mid
            self.h, self.lo, self.c = max(self.h, mid), min(self.lo, mid), mid
        if self.n < self.per_bar:
            return None
        bar = None if math.isnan(self.o) else (self.o, self.h, self.lo, self.c)
        self.n = 0
        self.o = self.h = self.lo = self.c = math.nan
        return bar


def bar_features(mids, per_bar: int) -> list[BarFeatures]:
    """Features of every bar in a sequence of decision-time mids (ticks; NaN when one-sided)."""
    bb, fe, out = BarBuilder(per_bar), FeatureEngine(), []
    for m in mids:
        bar = bb.add(float(m))
        if bar is not None:
            out.append(fe.update(*bar))
    return out


# ---------------------------------------------------------------- regime detector


def kmeans(x: np.ndarray, k: int, n_init: int = 10, seed: int = 42, iters: int = 300):
    """Lloyd's algorithm with k-means++ starts; the restart with the least inertia wins."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.array(c)
        for _ in range(iters):
            lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(new, c):
                break
            c = new
        inertia = ((x - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, c)
    return best[1]


REGIMES = ("calm", "trend_up", "trend_down", "volatile", "stressed")
HIGH_ENERGY = {"volatile", "stressed"}


@dataclass
class RegimeModel:
    """Standardised k-means centroids and the FYP's names for them."""

    mean: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray  # standardised
    names: tuple[str, ...]

    @classmethod
    def fit(cls, x: np.ndarray, k: int = 5, seed: int = 42) -> RegimeModel:
        mean, scale = x.mean(0), x.std(0)
        scale = np.where(scale > 0, scale, 1.0)
        c = kmeans((x - mean) / scale, k, seed=seed)
        raw = c * scale + mean
        order = np.argsort(raw[:, 0])  # by vol_20
        names = [""] * k
        names[order[0]] = "calm"
        a, b = order[1], order[2]
        names[a], names[b] = ("trend_up", "trend_down") if raw[a, 2] >= raw[b, 2] else \
            ("trend_down", "trend_up")
        a, b = order[3], order[4]
        names[a], names[b] = ("volatile", "stressed") if raw[a, 1] <= raw[b, 1] else \
            ("stressed", "volatile")
        return cls(mean, scale, c, tuple(names))

    def label(self, v: np.ndarray) -> str:
        z = (v - self.mean) / self.scale
        return self.names[int(np.argmin(((self.centroids - z) ** 2).sum(1)))]


class RegimeDetector:
    """Provisional k-means label, CUSUM gate into high-energy states, persistence."""

    def __init__(self, model: RegimeModel | None, persistence: int = 2) -> None:
        self.model, self.persistence = model, persistence
        self.cusum = CUSUM()
        self.current = "calm"
        self._prov, self._count = "calm", 0

    def update(self, f: BarFeatures) -> str:
        return self.confirm(self.model.label(f.regime_vector()), f.ret, f.sigma_60)

    def confirm(self, prov: str, ret: float, sigma: float) -> str:
        """The FYP's confirm_transition: the confirmed label after one bar."""
        fired = self.cusum.update(ret, sigma)
        if prov in HIGH_ENERGY and self.current not in HIGH_ENERGY and not fired:
            self._prov, self._count = self.current, 0
            return self.current
        if prov == self._prov:
            self._count += 1
        else:
            self._prov, self._count = prov, 1
        if self._count >= self.persistence:
            self.current = prov
            if prov not in HIGH_ENERGY:
                self.cusum.reset()
        return self.current


class AnomalyDetector:
    """none / moderate / severe from the two z-scores, with persistence."""

    def __init__(self, moderate: float = 2.0, severe: float = 3.0, persist_moderate: int = 1,
                 persist_severe: int = 1) -> None:
        self.mod, self.sev = moderate, severe
        self.p_mod, self.p_sev = persist_moderate, persist_severe
        self._n_mod = self._n_sev = 0

    def update(self, f: BarFeatures, regime: str) -> str:
        return self.level(f.z_ret, f.z_range, regime)

    def level(self, z_ret: float, z_range: float, regime: str) -> str:
        """The FYP's score_z (no event window)."""
        z = max(abs(z_ret), abs(z_range))
        self._n_sev = self._n_sev + 1 if z >= self.sev else 0
        self._n_mod = self._n_mod + 1 if z >= self.mod else 0
        need = self.p_sev if regime == "stressed" else self.p_mod
        if self._n_sev >= need:
            return "severe"
        if self._n_mod >= self.p_mod:
            return "moderate"
        return "none"


# ---------------------------------------------------------------- policy state machine


class Action(StrEnum):
    ALLOW = "allow"
    THROTTLE = "throttle"
    PAUSE = "pause"


@dataclass
class Policy:
    """The FYP's PolicyState and DecisionPolicy rules, without the hedge path."""

    throttle_size: float = 0.33
    throttle_bars: int = 5
    pause_bars: int = 10
    action: Action = Action.ALLOW
    bars_in_state: int = 0
    cooldown_throttle: int = 0
    cooldown_pause: int = 0
    clear_bars: int = 0
    reason: str = "INIT"

    def _go(self, new: Action, reason: str) -> None:
        if new == self.action:
            return
        if self.action == Action.THROTTLE:
            self.cooldown_throttle = self.throttle_bars
        elif self.action == Action.PAUSE:
            self.cooldown_pause = self.pause_bars
        self.action, self.bars_in_state, self.reason = new, 0, reason

    def decide(self, regime: str, anomaly: str) -> Action:
        self.bars_in_state += 1
        self.cooldown_throttle = max(0, self.cooldown_throttle - 1)
        self.cooldown_pause = max(0, self.cooldown_pause - 1)
        clear = regime in ("calm", "trend_up", "trend_down") and anomaly == "none"
        self.clear_bars = self.clear_bars + 1 if clear else 0

        if self.action == Action.THROTTLE and self.bars_in_state >= self.throttle_bars:
            self._go(Action.ALLOW, "THROTTLE_CLEARED" if self.clear_bars >= 3
                     else "THROTTLE_EXPIRED")
            return self.action
        if self.action == Action.PAUSE and self.bars_in_state >= self.pause_bars:
            self._go(Action.ALLOW, "PAUSE_CLEARED" if self.clear_bars >= 3 else "PAUSE_EXPIRED")
            return self.action
        if (regime == "stressed" and anomaly == "severe" and self.action != Action.PAUSE
                and self.cooldown_pause == 0):
            self._go(Action.PAUSE, "SEVERE_ANOMALY_STRESSED_REGIME")
            return self.action
        if (self.action == Action.THROTTLE and regime == "stressed" and anomaly == "severe"
                and self.bars_in_state >= 2 and self.cooldown_pause == 0):
            self._go(Action.PAUSE, "THROTTLE_ESCALATED_TO_PAUSE")
            return self.action
        if (regime in ("volatile", "stressed", "trend_up", "trend_down")
                and anomaly in ("moderate", "severe") and self.action == Action.ALLOW
                and self.cooldown_throttle == 0):
            self._go(Action.THROTTLE, "VOLATILITY_MODERATE_EXPANSION")
            return self.action
        if self.clear_bars >= 3 and self.action != Action.ALLOW:
            self._go(Action.ALLOW, "CLEAR_CONDITIONS")
        return self.action


# ---------------------------------------------------------------- the overlay


@dataclass
class BarRecord:
    ts: int  # ns at the bar's close
    regime: str
    anomaly: str
    action: str
    killed: bool


@dataclass
class RiskOverlay:
    """Wraps a strategy (maker.Strategy) and edits its quotes. Build a fresh one per session.

    regimes=None turns the regime and anomaly policy off (the limits still apply)."""

    strategy: object
    regimes: RegimeModel | None = None
    bar_s: float = 5.0
    step_s: float = 0.1
    inventory_cap: int | None = 500
    max_drawdown: float | None = None  # price units ($1 = 10,000)
    bars: list[BarRecord] = field(default_factory=list)
    killed_at: int | None = None

    def __post_init__(self) -> None:
        self._bb = BarBuilder(round(self.bar_s / self.step_s))
        self._fe = FeatureEngine()
        self._reg = RegimeDetector(self.regimes) if self.regimes is not None else None
        self._anom = AnomalyDetector()
        self.policy = Policy()
        self._peak = 0.0

    def warm_up(self, mids) -> None:
        """Feeds earlier decision-time mids (ticks) through the features and detectors, so the
        windows are full from the first bar. The policy starts fresh."""
        for m in mids:
            bar = self._bb.add(float(m))
            if bar is not None:
                f = self._fe.update(*bar)
                if self._reg is not None:
                    self._anom.update(f, self._reg.update(f))
        self._bb = BarBuilder(self._bb.per_bar)

    def quote(self, v: MarketView) -> Quotes:
        q = self.strategy.quote(v)
        mid_ticks = v.mid / v.tick if v.mid is not None else None
        bar = self._bb.add(mid_ticks)
        if bar is not None:
            f = self._fe.update(*bar)
            regime = anomaly = "-"
            if self._reg is not None:
                regime = self._reg.update(f)
                anomaly = self._anom.update(f, regime)
                if f.ready:
                    self.policy.decide(regime, anomaly)
            self.bars.append(BarRecord(v.now, regime, anomaly, self.policy.action.value,
                                       self.killed_at is not None))
        # Drawdown kill switch on the marked PnL.
        if self.max_drawdown is not None and v.mid is not None and self.killed_at is None:
            pnl = v.cash + v.inventory * v.mid
            self._peak = max(self._peak, pnl)
            if self._peak - pnl >= self.max_drawdown:
                self.killed_at = v.now
        reduce_only = (self.killed_at is not None or self.policy.action == Action.PAUSE
                       or (self.inventory_cap is not None
                           and abs(v.inventory) >= self.inventory_cap))
        bid, ask, bs, az = q.bid, q.ask, q.bid_size, q.ask_size
        if self.policy.action == Action.THROTTLE:
            bs = max(1, round(bs * self.policy.throttle_size))
            az = max(1, round(az * self.policy.throttle_size))
        if reduce_only:
            bid = bid if v.inventory < 0 else None
            ask = ask if v.inventory > 0 else None
        # Never quote more than it takes to get flat on the reducing side.
        if reduce_only and bid is not None:
            bs = min(bs, -v.inventory)
        if reduce_only and ask is not None:
            az = min(az, v.inventory)
        return Quotes(bid, ask, bs, az)
