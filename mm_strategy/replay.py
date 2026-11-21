"""Market making against a recorded tape, with virtual orders (Project 3, step 5).

A tape is a stream of LOBSTER messages: Nasdaq's AAPL day, or a synthetic session written out
by mm_strategy.tape. The C++ replay (orderbook.lobster.ArrayReplay) rebuilds the book from it
message by message, as in Project 2's LOBSTER gate, and makers trade against that book.

Their orders never enter it, so the market never sees them or reacts to them. That is the
blind spot of any replay, and step 5 measures it on the synthetic market, where the same
strategy can also trade live. Each virtual order keeps its place in its queue:
- it joins behind everything already resting at its price: the queue ahead starts at that
  level's displayed size;
- an order that was there when it joined shrinks the queue ahead by whatever is cancelled or
  executed from it. Orders that arrive later join behind it and never count;
- an execution at its price fills it only once nothing is left ahead. With nothing ahead, the
  order executed in the tape must have been behind it, so in a world with the maker in it the
  aggressor would have reached the maker first;
- an execution at a worse price (a resting buy below the maker's bid, say) fills it at once,
  from an aggressor who passed its price on the way;
- a hidden execution fills it at a worse price, or at its own price with nothing ahead
  (displayed orders have priority over hidden ones at a price).
A new limit order priced through the virtual order does not fill it by default
(cross_fills=False). In the tape that order was placed against a book without the maker in
it. On the synthetic market, where live trading is the ground truth, counting such orders as
fills makes the replay markedly more optimistic (experiments.blind_spot measures by how much).

Fills are passive, at the virtual order's price, and earn the maker rebate. One aggressor can
fill the maker and still trade everything the tape says it traded; that liquidity is counted
twice, which is the price of not letting the maker change the tape.

Order management mirrors maker.MarketMaker: a size cut keeps the queue place, a partly filled
order at an unchanged price is left alone, a new price goes to the back of the new queue, and
quotes stay at least a tick clear of the opposite best (post-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mm_strategy.accounting import PnL, split_pnl
from mm_strategy.maker import Fees, Fill, MarketView
from mm_strategy.market import NS
from mm_strategy.tape import (
    CANCEL,
    DELETE,
    EXECUTE,
    EXECUTE_HIDDEN,
    SUBMIT,
    Tape,
)
from orderbook import Side
from orderbook.lobster import ArrayReplay, DayReader, Event, Message, Snapshot

NO_BID, NO_ASK = np.iinfo(np.int64).min, np.iinfo(np.int64).max
_SIDE = {1: Side.Buy, -1: Side.Sell}
# How a virtual fill came about.
QUEUE, THROUGH, HIDDEN, CROSSED = 1, 2, 3, 4
FILL_KINDS = {QUEUE: "at its price, queue ahead gone", THROUGH: "traded through its price",
              HIDDEN: "hidden execution", CROSSED: "crossed by a new limit order"}


@dataclass
class VirtualOrder:
    """One resting virtual order. side: +1 a bid, -1 an ask (LOBSTER's direction)."""

    side: int
    price: int
    qty: int
    ahead: int  # displayed shares ahead of it in its queue
    behind: set[int] = field(default_factory=set)  # ids that joined its price after it
    cross_fills: bool = False  # whether a new order priced through it fills it
    kind: int = 0  # how the last fill came about (FILL_KINDS)

    def on_message(self, event: int, oid: int, size: int, price: int, d: int) -> int:
        """Updates the queue for one message, before the book applies it, and returns the
        shares this order fills."""
        s = self.side
        if event == SUBMIT:
            if d == s:
                if price == self.price:
                    self.behind.add(oid)
                return 0
            if self.cross_fills and s * price <= s * self.price:
                return self._fill(size, CROSSED)
            return 0
        if d != s:
            return 0
        if event == EXECUTE:
            if s * price < s * self.price:
                return self._fill(size, THROUGH)
            if price == self.price:
                if oid in self.behind and self.ahead == 0:
                    return self._fill(size, QUEUE)
                self.ahead = max(0, self.ahead - size)
            return 0
        if event in (CANCEL, DELETE):
            if price == self.price and oid not in self.behind:
                self.ahead = max(0, self.ahead - size)
            return 0
        if event == EXECUTE_HIDDEN:
            if s * price < s * self.price or (price == self.price and self.ahead == 0):
                return self._fill(size, HIDDEN)
        return 0

    def _fill(self, size: int, kind: int) -> int:
        q = min(self.qty, size)
        self.qty -= q
        self.kind = kind
        return q


def level_qty(book, side: Side, price: int) -> int:
    """Displayed shares resting at one price."""
    return sum(book.find(i).qty for i in book.queue(side, price))


@dataclass
class ReplayMaker:
    """A maker trading virtually in [start, end] (ns on the tape's clock). It wakes every
    `requote_ms` from start, the first time at start + requote_ms, as maker.MarketMaker does."""

    strategy: object
    start: int
    end: int
    requote_ms: int = 100
    fees: Fees = field(default_factory=Fees)
    view_levels: int = 5
    tick: int = 100
    cross_fills: bool = False  # see the module docstring
    inventory: int = 0
    cash: int = 0
    fees_paid: int = 0
    fills: list[Fill] = field(default_factory=list)
    fill_mid2: list[int] = field(default_factory=list)  # bid + ask before each fill's timestamp
    fill_kind: list[int] = field(default_factory=list)
    samples: list[tuple[int, int, int]] = field(default_factory=list)
    orders_sent: int = 0

    def __post_init__(self) -> None:
        self.orders: dict[int, VirtualOrder | None] = {1: None, -1: None}
        self.next_wake = self.start + self.requote_ms * 1_000_000
        self.mid2_end: int | None = None

    def on_message(self, event, oid, size, price, d, aggressor, ns, mid2_before) -> None:
        for s in (1, -1):
            o = self.orders[s]
            if o is None:
                continue
            q = o.on_message(event, oid, size, price, d)
            if q:
                fee = -self.fees.maker_rebate * q
                self.inventory += s * q
                self.cash -= s * o.price * q + fee
                self.fees_paid += fee
                self.fills.append(Fill(int(ns), s, int(o.price), int(q), True, int(aggressor)))
                self.fill_mid2.append(int(mid2_before))
                self.fill_kind.append(o.kind)
                if o.qty == 0:
                    self.orders[s] = None

    def wake(self, book, now: int) -> None:
        bids = tuple((lv.price, lv.qty) for lv in book.depth(Side.Buy, self.view_levels))
        asks = tuple((lv.price, lv.qty) for lv in book.depth(Side.Sell, self.view_levels))
        bid = bids[0][0] if bids else None
        ask = asks[0][0] if asks else None
        mid = (bid + ask) / 2 if bid is not None and ask is not None else None
        q = self.strategy.quote(MarketView(now, self.end, bid, ask, mid, self.inventory,
                                           self.tick, bids, asks))
        self._place(book, 1, q.bid, q.bid_size, ask)
        self._place(book, -1, q.ask, q.ask_size, bid)
        self.samples.append((now, self.inventory, self.cash))
        self.next_wake = now + self.requote_ms * 1_000_000

    def _place(self, book, s: int, price, size: int, opposite) -> None:
        if price is not None and opposite is not None:  # post-only: a tick clear
            price = min(price, opposite - self.tick) if s == 1 else max(price, opposite + self.tick)
        if price is not None and price <= 0:
            price = None
        o = self.orders[s]
        if price is None or size <= 0:
            self.orders[s] = None
            return
        if o is not None and o.price == price:
            o.qty = min(o.qty, size)  # a cut keeps the place; a partial fill is left alone
            return
        self.orders_sent += 1
        self.orders[s] = VirtualOrder(s, int(price), int(size), level_qty(book, _SIDE[s], price),
                                      cross_fills=self.cross_fills)

    def finish(self, mid2_end: int) -> None:
        self.orders = {1: None, -1: None}
        self.mid2_end = mid2_end

    # ---- results

    def fill_arrays(self) -> dict[str, np.ndarray]:
        f = self.fills
        return {
            "ts": np.array([x.ts for x in f], dtype=np.int64),
            "side": np.array([x.side for x in f], dtype=np.int64),
            "price": np.array([x.price for x in f], dtype=np.int64),
            "qty": np.array([x.qty for x in f], dtype=np.int64),
            "passive": np.ones(len(f), dtype=bool),
            "counterparty": np.array([x.counterparty for x in f], dtype=np.int64),
            "mid2": np.array(self.fill_mid2, dtype=np.int64),
            "kind": np.array(self.fill_kind, dtype=np.int64),
        }

    def pnl(self) -> PnL:
        f = self.fill_arrays()
        return split_pnl(f["side"], f["qty"], f["price"], f["mid2"], self.mid2_end, self.cash,
                         self.inventory, self.fees_paid)


class TapeSource:
    """A synthetic session's tape (mm_strategy.tape.Tape). The book starts empty."""

    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.seed = Snapshot()
        self.seed_ns = 0

    def __iter__(self):
        t = self.tape
        empty = Snapshot()
        for i in range(len(t)):
            d = int(t.side[i])
            m = Message(ns=int(t.ns[i]), event=Event(int(t.event[i])), id=int(t.id[i]),
                        size=int(t.size[i]), price=int(t.price[i]), side=_SIDE[d])
            yield m, empty, int(t.event[i]), d, int(t.aggressor[i])


class LobsterSource:
    """A LOBSTER day. As in the Phase 2 gate, the book is seeded from the snapshot after the
    first message, and replay starts at the second."""

    def __init__(self, message_path: str, levels: int = 10) -> None:
        self.reader = DayReader(message_path, levels=levels)
        m0, s0 = next(self.reader)
        self.seed, self.seed_ns = s0, m0.ns

    def __iter__(self):
        for m, s in self.reader:
            yield m, s, int(m.event), 1 if m.side == Side.Buy else -1, 0


class TapeReplay:
    """Replays a source through the C++ book and drives virtual makers against it. Call run()
    repeatedly with increasing `until`; makers can be added for later windows."""

    def __init__(self, source, levels: int = 10, tick: int = 100) -> None:
        self.rp = ArrayReplay(levels=levels, tick=tick, digest_events=False)
        self.rp.seed(source.seed, source.seed_ns)
        self.book = self.rp.book
        self._it = iter(source)
        self._next = next(self._it, None)
        b, a = self.book.best_bid(), self.book.best_ask()
        self._q = [(source.seed_ns, NO_BID if b is None else b, NO_ASK if a is None else a)]
        self.messages = 0
        self._last_ns = source.seed_ns
        # bid + ask of the last two-sided book (one-sided moments are skipped, as in live
        # accounting), and its value as the last earlier timestamp ended.
        self._two = self._mid2() or 0
        self._mid2_prev = self._two

    def _mid2(self) -> int | None:
        _, b, a = self._q[-1]
        return b + a if b != NO_BID and a != NO_ASK else None

    def run(self, makers: list[ReplayMaker], until: int) -> None:
        """Processes every message with ns <= until, waking each maker on its clock (a maker
        sees the book after every message up to and including its wake time)."""
        live = [m for m in makers if m.next_wake <= m.end]
        while True:
            wake = min((m.next_wake for m in live), default=None)
            nxt = self._next
            if nxt is not None and nxt[0].ns <= until and (wake is None or nxt[0].ns <= wake):
                msg, snap, ev, d, agg = nxt
                ns = msg.ns
                if ns != self._last_ns:  # fills are measured against the mid before the
                    self._mid2_prev = self._two  # whole aggressive order, as live
                    self._last_ns = ns
                if live:
                    oid, size, price = msg.id, msg.size, msg.price
                    for m in live:
                        m.on_message(ev, oid, size, price, d, agg, ns, self._mid2_prev)
                self.rp.step(msg, snap)
                b, a = self.book.best_bid(), self.book.best_ask()
                self._q.append((ns, NO_BID if b is None else b, NO_ASK if a is None else a))
                if b is not None and a is not None:
                    self._two = b + a
                self.messages += 1
                self._next = next(self._it, None)
            elif wake is not None and wake <= until:
                for m in live:
                    if m.next_wake == wake:
                        m.wake(self.book, wake)
                for m in [m for m in live if m.next_wake > m.end]:
                    m.finish(self._two)
                live = [m for m in live if m.next_wake <= m.end]
            else:
                break

    def quotes(self) -> dict[str, np.ndarray]:
        """Best bid and ask after every message (and at the seed): ts, bid, ask."""
        a = np.array(self._q, dtype=np.int64)
        return {"ts": a[:, 0], "bid": a[:, 1], "ask": a[:, 2]}


def mid_at(quotes: dict[str, np.ndarray], ts, tick: int) -> np.ndarray:
    """Mid (ticks) in force at each time: the last quote at or before it; NaN if one-sided."""
    two = (quotes["bid"] != NO_BID) & (quotes["ask"] != NO_ASK)
    mid = np.where(two, (quotes["bid"].astype(float) + quotes["ask"].astype(float)) / 2, np.nan)
    i = np.searchsorted(quotes["ts"], np.asarray(ts, dtype=np.int64), side="right") - 1
    return np.where(i >= 0, mid[np.maximum(i, 0)], np.nan) / tick


def markouts(f: dict[str, np.ndarray], quotes, end: int, tick: int,
             horizons_s: tuple[float, ...]) -> dict:
    """Volume-weighted markouts of a maker's fills in ticks per share, overall and by
    counterparty, as accounting.maker_markouts computes them for a live session."""
    out: dict = {}
    groups = {"all": np.ones(len(f["ts"]), bool)}
    for c in np.unique(f["counterparty"]):
        groups[int(c)] = f["counterparty"] == c
    for name, sel in groups.items():
        row = {}
        for h in horizons_s:
            t_h = f["ts"] + int(h * NS)
            m = f["side"] * (mid_at(quotes, t_h, tick) - f["price"] / tick)
            ok = sel & (t_h <= end) & np.isfinite(m)
            w = f["qty"][ok]
            row[h] = float((m[ok] * w).sum() / w.sum()) if w.sum() else float("nan")
        out[name] = row
    return out
