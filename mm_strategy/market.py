"""A synthetic market for Project 3: a hidden fair value and the traders who chase it.

The fair value follows geometric Brownian motion, simulated with Project 1's exact step
(mc_engine.gbm.paths) on a 10 ms grid. Nobody in the market sees it except the informed traders.
Three kinds of agents send orders into the C++ book (orderbook.ArrayOrderBook). Each is a
Poisson process with its own random stream, so adding a participant later leaves the others'
arrival times unchanged:

  noise limit orders   rest k ticks behind the opposite best quote, P(k) ~ k^-alpha (k >= 1, so
                       never marketable); each is cancelled after an exponential lifetime
                       unless it fills first
  noise market orders  buy or sell at random
  informed traders     see the fair value V and take liquidity when the quotes are stale:
                       sweep every ask at least `edge` ticks below V (every bid at least
                       `edge` ticks above it), up to `informed_max` shares

Noise orders are placed relative to the book, so on their own they would let the price wander.
The informed traders pull it back to V, and in doing so they pick off resting orders. That is
the adverse selection a market maker in this market must survive.

Prices are integer LOBSTER units ($1 = 10,000) on a one-cent tick (100 units), and time is in
nanoseconds from the open, as in the AAPL replay. A session is reproducible from its seed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from mc_engine import gbm
from mm_strategy.tape import TapeRecorder
from orderbook import ArrayOrderBook, OrderType, Side, Trade
from orderbook.analytics import DepthProfile, OrderFlowImbalance, SpreadDistribution

NS = 1_000_000_000
DOLLAR = 10_000  # LOBSTER price units per dollar
TRADING_YEAR_HOURS = 252 * 6.5

# Agent codes, carried in the top bits of every order id.
NOISE_LIMIT, NOISE_MARKET, INFORMED, MAKER = 1, 2, 3, 4
AGENT_NAMES = {NOISE_LIMIT: "noise limit", NOISE_MARKET: "noise market", INFORMED: "informed",
               MAKER: "market maker"}
ID_SHIFT = 40


def order_id(agent: int, n: int) -> int:
    return (agent << ID_SHIFT) | n


def agent_of(oid: int | np.ndarray):
    return oid >> ID_SHIFT


@dataclass(frozen=True)
class MarketParams:
    """One session's market. The defaults give a book about two ticks wide, quoted on both sides
    99.9% of the time, whose mid stays a median two ticks from the fair value."""

    s0: float = 100.0  # fair value at the open, dollars
    sigma: float = 0.30  # annualised volatility of the fair value
    hours: float = 1.0
    tick: int = 100  # price units per tick: one cent
    fv_step_ms: int = 10
    # Noise limit orders.
    limit_rate: float = 8.0  # per second
    alpha: float = 0.7  # placement: P(k ticks behind the opposite best) ~ k^-alpha
    max_ticks: int = 25
    mean_life_s: float = 4.0
    # Noise market orders.
    market_rate: float = 1.0  # per second
    # Informed traders.
    informed_rate: float = 0.7  # per second
    edge_ticks: int = 2
    informed_max: int = 2_000  # shares an informed trader will take in one sweep
    # Order sizes: lot x Geometric(p), so the mean is lot / p.
    lot: int = 100
    size_p: float = 0.5
    max_lots: int = 10
    # A stress episode (step 6): from stress_start_s for stress_len_s seconds the fair value's
    # volatility is multiplied by stress_vol, and informed traders arrive stress_informed times
    # as often. Off by default (stress_len_s = 0), which leaves every session as it was.
    stress_start_s: float = 0.0
    stress_len_s: float = 0.0
    stress_vol: float = 1.0
    stress_informed: float = 1.0

    @property
    def duration_ns(self) -> int:
        return int(self.hours * 3600 * NS)


@dataclass
class Session:
    """What one simulated session produced."""

    params: MarketParams
    seed: int
    fair_value: np.ndarray  # price units on the fv_step_ms grid, fair_value[0] at t = 0
    trades: dict  # numpy columns: ts, price, qty, taker_side (+1 buy / -1 sell), taker, maker
    quotes: dict  # every quote update (OrderFlowImbalance.records())
    spread: SpreadDistribution
    depth: DepthProfile
    ofi: OrderFlowImbalance
    digest: int  # FNV-1a of the book's whole event stream
    calls: dict = field(default_factory=dict)  # engine calls by agent
    tape: object = None  # tape.Tape of LOBSTER messages, if simulate(record_tape=True)

    def fair_value_at(self, ts: np.ndarray | int) -> np.ndarray:
        step = self.params.fv_step_ms * 1_000_000
        i = np.minimum(np.asarray(ts, dtype=np.int64) // step, len(self.fair_value) - 1)
        return self.fair_value[i]

    def mid_at(self, ts: np.ndarray | int) -> np.ndarray:
        """Mid price (units) in force at each time: the last quote update at or before it."""
        q = self.quotes
        two = (q["bid"] != np.iinfo(np.int64).min) & (q["ask"] != np.iinfo(np.int64).max)
        mid = np.where(two, (q["bid"].astype(float) + q["ask"].astype(float)) / 2, np.nan)
        i = np.searchsorted(q["ts"].astype(np.int64), np.asarray(ts, dtype=np.int64),
                            side="right") - 1
        return np.where(i >= 0, mid[np.maximum(i, 0)], np.nan)


class _Placement:
    """Tick distances k = 1..K drawn with probability proportional to k^-alpha."""

    def __init__(self, alpha: float, max_ticks: int) -> None:
        w = np.arange(1, max_ticks + 1, dtype=float) ** -alpha
        self.cdf = np.cumsum(w / w.sum())

    def draw(self, rng: np.random.Generator) -> int:
        return int(np.searchsorted(self.cdf, rng.random(), side="right")) + 1


def fair_value_path(p: MarketParams, rng: np.random.Generator) -> np.ndarray:
    """GBM fair value in price units on the fv_step_ms grid (Project 1's exact step). During a
    stress episode the volatility is stress_vol times higher; the step stays exact."""
    n_steps = int(round(p.hours * 3600 * 1000 / p.fv_step_ms))
    years = p.hours / TRADING_YEAR_HOURS
    if p.stress_len_s <= 0 or p.stress_vol == 1.0:
        path = gbm.paths(p.s0, 0.0, p.sigma, years, n_steps, 1, rng)[0]
        return path * DOLLAR
    dt = years / n_steps
    t = np.arange(n_steps) * p.fv_step_ms / 1000  # start of each step, seconds
    stressed = (t >= p.stress_start_s) & (t < p.stress_start_s + p.stress_len_s)
    sig = np.where(stressed, p.sigma * p.stress_vol, p.sigma)
    steps = -0.5 * sig**2 * dt + sig * np.sqrt(dt) * rng.standard_normal(n_steps)
    return p.s0 * np.exp(np.r_[0.0, np.cumsum(steps)]) * DOLLAR


def simulate(p: MarketParams, seed: int, participant=None, record_tape: bool = False) -> Session:
    """Runs one session. `participant`, if given, is driven by the same event loop (the market
    maker of step 2 onwards). It gets `on_start(sim)`, then `on_event(sim, payload)` for each
    event it scheduled through `sim.schedule(t, sim.PARTICIPANT, payload)`, and
    `on_fill(sim, trade)` for every trade with one of its own (MAKER) orders in it.
    record_tape=True also writes the session out as LOBSTER messages (Session.tape; step 5)."""
    return _Simulator(p, seed, participant, record_tape).run()


class _Simulator:
    # Event kinds, in tie-break order at equal times (cancels before new arrivals).
    CANCEL, LIMIT, MARKET, INFORMED_ARRIVAL, PARTICIPANT = range(5)

    def __init__(self, p: MarketParams, seed: int, participant, record_tape: bool = False) -> None:
        self.p = p
        self.seed = seed
        ss = np.random.SeedSequence(seed)
        fv_ss, lim_ss, mkt_ss, inf_ss, part_ss = ss.spawn(5)
        self.rng_limit = np.random.default_rng(lim_ss)
        self.rng_market = np.random.default_rng(mkt_ss)
        self.rng_informed = np.random.default_rng(inf_ss)
        self.rng_participant = np.random.default_rng(part_ss)
        self.fv = fair_value_path(p, np.random.default_rng(fv_ss))
        self.fv_step = p.fv_step_ms * 1_000_000
        self.placement = _Placement(p.alpha, p.max_ticks)
        self.book = ArrayOrderBook(tick_size=p.tick, expected_orders=1 << 16)
        self.spread = SpreadDistribution(p.tick)
        self.depth = DepthProfile(p.tick, 10)
        self.ofi = OrderFlowImbalance(window=100, record=True)
        for listener in (self.spread, self.depth, self.ofi):
            self.book.attach(listener)
        self.heap: list[tuple[int, int, int, object]] = []
        self.seq = 0
        self.counter = dict.fromkeys(AGENT_NAMES, 0)
        self.calls = dict.fromkeys(AGENT_NAMES, 0)
        self.trades: list[tuple[int, int, int, int, int, int]] = []
        self.now = 0
        self.participant = participant
        self.recorder = TapeRecorder() if record_tape else None

    # ---- scheduling

    def schedule(self, t: int, kind: int, payload: object = None) -> None:
        self.seq += 1
        heapq.heappush(self.heap, (t, kind, self.seq, payload))

    def _next_arrival(self, rate: float, rng: np.random.Generator, kind: int) -> None:
        self.schedule(self.now + int(rng.exponential(NS / rate)) + 1, kind)

    # ---- market state

    def fair_value(self) -> float:
        return float(self.fv[min(self.now // self.fv_step, len(self.fv) - 1)])

    def size(self, rng: np.random.Generator) -> int:
        return self.p.lot * min(int(rng.geometric(self.p.size_p)), self.p.max_lots)

    def new_id(self, agent: int) -> int:
        self.counter[agent] += 1
        return order_id(agent, self.counter[agent])

    def call(self, agent: int, fn, *args):
        """One engine call on behalf of `agent`; records the trades it caused and tells the
        participant about the ones it was in."""
        self.calls[agent] += 1
        r = fn(*args)
        events = self.book.take_events()
        if self.recorder is not None:
            self.recorder.record(events, self.book)
        for e in events:
            if isinstance(e, Trade):
                side = 1 if e.taker_side == Side.Buy else -1
                self.trades.append((e.ts, e.price, e.qty, side, e.taker, e.maker))
                if self.participant is not None and MAKER in (agent_of(e.maker),
                                                              agent_of(e.taker)):
                    self.participant.on_fill(self, e)
        return r

    # ---- agents

    def _noise_limit(self) -> None:
        rng, p = self.rng_limit, self.p
        side = Side.Buy if rng.random() < 0.5 else Side.Sell
        k = self.placement.draw(rng)
        bid, ask = self.book.best_bid(), self.book.best_ask()
        if side == Side.Buy:
            ref = ask if ask is not None else (bid + p.tick if bid is not None else None)
            price = (ref if ref is not None else self._fv_tick()) - k * p.tick
        else:
            ref = bid if bid is not None else (ask - p.tick if ask is not None else None)
            price = (ref if ref is not None else self._fv_tick()) + k * p.tick
        oid = self.new_id(NOISE_LIMIT)
        if price > 0:
            self.call(NOISE_LIMIT, self.book.submit, oid, side, price, self.size(rng),
                      OrderType.Limit, self.now)
            life = int(rng.exponential(p.mean_life_s * NS)) + 1
            self.schedule(self.now + life, self.CANCEL, oid)
        self._next_arrival(p.limit_rate, rng, self.LIMIT)

    def _noise_market(self) -> None:
        rng = self.rng_market
        side = Side.Buy if rng.random() < 0.5 else Side.Sell
        self.call(NOISE_MARKET, self.book.submit, self.new_id(NOISE_MARKET), side, 0,
                  self.size(rng), OrderType.Market, self.now)
        self._next_arrival(self.p.market_rate, rng, self.MARKET)

    def in_stress(self) -> bool:
        p, t = self.p, self.now / NS
        return p.stress_len_s > 0 and p.stress_start_s <= t < p.stress_start_s + p.stress_len_s

    def _informed_rate_max(self) -> float:
        p = self.p
        return p.informed_rate * (p.stress_informed if p.stress_len_s > 0 else 1.0)

    def _informed(self) -> None:
        rng, p = self.rng_informed, self.p
        # A stress episode raises the arrival rate: arrivals come at the stressed rate and,
        # outside the episode, are kept with probability 1 / stress_informed (thinning).
        if p.stress_len_s > 0 and p.stress_informed != 1.0 and not self.in_stress():
            if rng.random() >= 1 / p.stress_informed:
                self._next_arrival(self._informed_rate_max(), rng, self.INFORMED_ARRIVAL)
                return
        v = self.fair_value()
        edge = p.edge_ticks * p.tick
        bid, ask = self.book.best_bid(), self.book.best_ask()
        qty = p.informed_max
        if ask is not None and v - ask >= edge:
            limit = int(math.floor((v - edge) / p.tick)) * p.tick  # buy everything up to here
            self.call(INFORMED, self.book.submit, self.new_id(INFORMED), Side.Buy, limit, qty,
                      OrderType.IOC, self.now)
        elif bid is not None and bid - v >= edge:
            limit = int(math.ceil((v + edge) / p.tick)) * p.tick
            self.call(INFORMED, self.book.submit, self.new_id(INFORMED), Side.Sell, limit, qty,
                      OrderType.IOC, self.now)
        self._next_arrival(self._informed_rate_max(), rng, self.INFORMED_ARRIVAL)

    def _fv_tick(self) -> int:
        return int(round(self.fair_value() / self.p.tick)) * self.p.tick

    # ---- loop

    def run(self) -> Session:
        p = self.p
        end = p.duration_ns
        self._next_arrival(p.limit_rate, self.rng_limit, self.LIMIT)
        self._next_arrival(p.market_rate, self.rng_market, self.MARKET)
        self._next_arrival(self._informed_rate_max(), self.rng_informed, self.INFORMED_ARRIVAL)
        if self.participant is not None:
            self.participant.on_start(self)
        while self.heap:
            t, kind, _, payload = heapq.heappop(self.heap)
            if t > end:
                break
            self.now = t
            if kind == self.CANCEL:
                self.call(NOISE_LIMIT, self.book.cancel, payload, t)
            elif kind == self.LIMIT:
                self._noise_limit()
            elif kind == self.MARKET:
                self._noise_market()
            elif kind == self.INFORMED_ARRIVAL:
                self._informed()
            else:
                self.participant.on_event(self, payload)
        cols = list(zip(*self.trades, strict=True)) if self.trades else [()] * 6
        names = ("ts", "price", "qty", "taker_side", "taker", "maker")
        dtypes = (np.int64, np.int64, np.int64, np.int8, np.uint64, np.uint64)
        trades = {n: np.array(c, dtype=d) for n, c, d in zip(names, cols, dtypes, strict=True)}
        return Session(
            params=p, seed=self.seed, fair_value=self.fv, trades=trades,
            quotes=self.ofi.records(), spread=self.spread, depth=self.depth, ofi=self.ofi,
            digest=self.book.digest, calls={AGENT_NAMES[a]: n for a, n in self.calls.items()},
            tape=self.recorder.tape() if self.recorder is not None else None,
        )
