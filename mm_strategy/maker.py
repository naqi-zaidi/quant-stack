"""A market maker inside the synthetic market (Project 3, step 2): quotes, fills, inventory.

The maker is a participant in the market's event loop (market.simulate). Every `requote_ms` it
asks its strategy for a bid and an ask, then moves its two resting orders there through the C++
book:
- a size cut at the same price is a modify, which keeps the order's place in the queue;
- a partly filled order at an unchanged price is left alone, keeping its place;
- a new price is a cancel-replace (modify).

Quotes are post-only, so the maker never takes liquidity. Its bid is kept at least one tick below
the best ask, and its ask one tick above the best bid. Fills reach it as they happen, inside
other traders' calls. It books them into inventory and cash at once, and reacts at its next
wake-up.

The strategy sees the market without the maker's own orders: the best bid and ask of everyone
else, and their mid. Accounting (mm_strategy.accounting) marks inventory at the book's mid.

Fees are illustrative and configurable: a rebate per share filled passively and a fee per share
taken (the maker never pays it, being post-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from mm_strategy.market import MAKER, NS, agent_of
from orderbook import OrderType, Side, Trade


@dataclass(frozen=True)
class Fees:
    """Per-share fees in price units ($1 = 10,000): a rebate earned on passive fills, a fee paid
    on aggressive ones."""

    maker_rebate: int = 20  # $0.0020 a share
    taker_fee: int = 30  # $0.0030 a share


@dataclass(frozen=True)
class MarketView:
    """What a strategy sees at a decision: the market without the maker's own orders."""

    now: int  # ns from the open
    end: int  # ns at the close
    bid: int | None  # best bid of everyone else, price units
    ask: int | None
    mid: float | None  # (bid + ask) / 2 when both exist
    inventory: int  # shares, + long
    tick: int


@dataclass(frozen=True)
class Quotes:
    """A strategy's answer: prices in units (None = don't quote that side) and sizes."""

    bid: int | None
    ask: int | None
    bid_size: int = 100
    ask_size: int = 100


class Strategy(Protocol):
    def quote(self, view: MarketView) -> Quotes: ...


@dataclass(frozen=True)
class FixedSpread:
    """The baseline: quote `half_spread` ticks either side of the mid, rounded away from it,
    `size` shares a side. Stop quoting the side that would push |inventory| past the limit."""

    half_spread: float = 1.0  # ticks
    size: int = 100
    max_inventory: int = 2_000

    def quote(self, v: MarketView) -> Quotes:
        if v.mid is None:
            return Quotes(None, None)
        h = self.half_spread * v.tick
        bid = math.floor((v.mid - h) / v.tick) * v.tick
        ask = math.ceil((v.mid + h) / v.tick) * v.tick
        return Quotes(
            bid if v.inventory + self.size <= self.max_inventory else None,
            ask if v.inventory - self.size >= -self.max_inventory else None,
            self.size,
            self.size,
        )


@dataclass(frozen=True)
class AvellanedaStoikov:
    """Avellaneda and Stoikov (2008), "High-frequency trading in a limit order book".

    With mid s, inventory q (in lots of `size` shares), risk aversion gamma and time to the
    close tau:

        reservation price  r = s - q * gamma * sigma^2 * tau
        total spread       delta_a + delta_b = gamma * sigma^2 * tau + (2/gamma) ln(1 + gamma/k)

    It quotes symmetrically about r: bid r - spread/2, ask r + spread/2, rounded away from r to
    the tick. Units are ticks and seconds: sigma in ticks per sqrt(s), k per tick (from fill
    intensity A exp(-k delta)), gamma per tick per lot. A long position pulls both quotes down,
    so the maker sells more and buys less. gamma = 0 is the no-skew limit: total spread 2/k
    about the mid. The inventory limit still applies as a hard backstop.
    """

    gamma: float
    sigma: float
    k: float
    size: int = 100
    max_inventory: int = 2_000

    def reservation_offset(self, q_lots: float, tau_s: float) -> float:
        """r - s, in ticks."""
        return -q_lots * self.gamma * self.sigma**2 * tau_s

    def total_spread(self, tau_s: float) -> float:
        """delta_a + delta_b, in ticks."""
        if self.gamma == 0:
            return 2 / self.k
        return (self.gamma * self.sigma**2 * tau_s
                + 2 / self.gamma * math.log1p(self.gamma / self.k))

    def quote(self, v: MarketView) -> Quotes:
        if v.mid is None:
            return Quotes(None, None)
        tau = max(v.end - v.now, 0) / NS
        r = v.mid / v.tick + self.reservation_offset(v.inventory / self.size, tau)
        half = self.total_spread(tau) / 2
        bid = math.floor(r - half) * v.tick
        ask = math.ceil(r + half) * v.tick
        return Quotes(
            bid if v.inventory + self.size <= self.max_inventory else None,
            ask if v.inventory - self.size >= -self.max_inventory else None,
            self.size,
            self.size,
        )


@dataclass
class Fill:
    ts: int
    side: int  # +1 the maker bought, -1 it sold
    price: int
    qty: int
    passive: bool
    counterparty: int  # agent code of the other side


@dataclass
class MarketMaker:
    """The participant. Pass it to market.simulate(params, seed, participant=MarketMaker(...))."""

    strategy: Strategy
    requote_ms: int = 100
    fees: Fees = field(default_factory=Fees)
    # Running state, readable after the session.
    inventory: int = 0
    cash: int = 0  # price units, rebates and fees included
    fees_paid: int = 0  # negative when rebates exceed fees
    fills: list[Fill] = field(default_factory=list)
    samples: list[tuple[int, int, int]] = field(default_factory=list)  # (ts, inventory, cash)
    orders_sent: int = 0
    rejects: int = 0

    def __post_init__(self) -> None:
        self._live: dict[Side, int | None] = {Side.Buy: None, Side.Sell: None}

    # ---- participant interface

    def on_start(self, sim) -> None:
        sim.schedule(self.requote_ms * 1_000_000, sim.PARTICIPANT, "wake")

    def on_event(self, sim, payload) -> None:
        self._requote(sim)
        self.samples.append((sim.now, self.inventory, self.cash))
        sim.schedule(sim.now + self.requote_ms * 1_000_000, sim.PARTICIPANT, "wake")

    def on_fill(self, sim, t: Trade) -> None:
        """Called by the market inside an engine call, for every trade with the maker in it."""
        mine_maker = agent_of(t.maker) == MAKER
        mine_taker = agent_of(t.taker) == MAKER
        if mine_maker and mine_taker:
            raise AssertionError("the maker traded with itself")
        taker_buys = t.taker_side == Side.Buy
        if mine_maker:
            side = -1 if taker_buys else 1
            fee = -self.fees.maker_rebate * t.qty
            other = int(agent_of(t.taker))
        else:
            side = 1 if taker_buys else -1
            fee = self.fees.taker_fee * t.qty
            other = int(agent_of(t.maker))
        self.inventory += side * t.qty
        self.cash -= side * t.price * t.qty + fee
        self.fees_paid += fee
        self.fills.append(Fill(int(t.ts), side, int(t.price), int(t.qty), mine_maker, other))

    # ---- quoting

    def _market_side(self, book, side: Side) -> int | None:
        """Best price of everyone else on one side (skipping a level that is only the maker)."""
        own = self._live[side]
        mine = book.find(own) if own is not None else None
        for lv in book.depth(side, 2):
            if mine is not None and lv.price == mine.price and lv.qty == mine.qty:
                continue
            return lv.price
        return None

    def _requote(self, sim) -> None:
        book, p = sim.book, sim.p
        bid, ask = self._market_side(book, Side.Buy), self._market_side(book, Side.Sell)
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        view = MarketView(sim.now, p.duration_ns, bid, ask, mid, self.inventory, p.tick)
        q = self.strategy.quote(view)
        self._place(sim, Side.Buy, q.bid, q.bid_size)
        self._place(sim, Side.Sell, q.ask, q.ask_size)

    def _place(self, sim, side: Side, price: int | None, size: int) -> None:
        book, tick, now = sim.book, sim.p.tick, sim.now
        # Post-only: never at or through the opposite best (which may be the maker's own order).
        if price is not None:
            if side == Side.Buy and (best := book.best_ask()) is not None:
                price = min(price, best - tick)
            elif side == Side.Sell and (best := book.best_bid()) is not None:
                price = max(price, best + tick)
            if price <= 0:
                price = None
        oid = self._live[side]
        live = book.find(oid) if oid is not None else None
        if live is None:
            self._live[side] = oid = None
        if price is None or size <= 0:
            if oid is not None:
                sim.call(MAKER, book.cancel, oid, now)
                self._live[side] = None
            return
        if live is not None:
            if live.price == price:
                if live.qty > size:  # cut to size: keeps its place in the queue
                    sim.call(MAKER, book.modify, oid, price, size, now)
                return  # unchanged, or partly filled: keep the place
            r = sim.call(MAKER, book.modify, oid, price, size, now)
            if r.accepted:
                return
            self.rejects += 1
            sim.call(MAKER, book.cancel, oid, now)
            self._live[side] = None
        new = sim.new_id(MAKER)
        self.orders_sent += 1
        r = sim.call(MAKER, book.submit, new, side, price, size, OrderType.PostOnly, now)
        if r.accepted:
            self._live[side] = new
        else:
            self.rejects += 1

    # ---- results

    def fill_arrays(self) -> dict[str, np.ndarray]:
        f = self.fills
        return {
            "ts": np.array([x.ts for x in f], dtype=np.int64),
            "side": np.array([x.side for x in f], dtype=np.int64),
            "price": np.array([x.price for x in f], dtype=np.int64),
            "qty": np.array([x.qty for x in f], dtype=np.int64),
            "passive": np.array([x.passive for x in f], dtype=bool),
            "counterparty": np.array([x.counterparty for x in f], dtype=np.int64),
        }


def sample_seconds(mm: MarketMaker) -> np.ndarray:
    """Sample times of the maker's (inventory, cash) record, in seconds."""
    return np.array([t for t, _, _ in mm.samples], dtype=float) / NS
