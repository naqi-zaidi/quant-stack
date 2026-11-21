"""The market-making harness of Project 3 step 2."""

from __future__ import annotations

import numpy as np
import pytest

from mm_strategy.accounting import decompose, inventory_stats, maker_fills_from_engine, reconcile
from mm_strategy.maker import FixedSpread, MarketMaker, MarketView, Quotes
from mm_strategy.market import MAKER, NS, MarketParams, agent_of, order_id, simulate
from orderbook import ArrayOrderBook, Side

SHORT = MarketParams(hours=0.05)


@pytest.fixture(scope="module")
def run():
    mm = MarketMaker(FixedSpread(half_spread=1))
    return simulate(SHORT, 21, participant=mm), mm


def test_books_reconcile_and_pnl_identity_holds(run) -> None:
    s, mm = run
    assert len(mm.fills) > 50
    assert reconcile(s, mm) == []
    pnl = decompose(s, mm)
    assert pnl.identity_holds
    assert pnl.net == pnl.spread + pnl.inventory - pnl.fees
    assert pnl.fees == -mm.fees.maker_rebate * pnl.volume  # all passive: rebates only


def test_maker_is_always_passive_and_never_self_trades(run) -> None:
    s, _ = run
    f = maker_fills_from_engine(s)
    assert f["passive"].all()
    both = (agent_of(s.trades["maker"].astype(np.int64)) == MAKER) & (
        agent_of(s.trades["taker"].astype(np.int64)) == MAKER
    )
    assert not both.any()


def test_same_seed_same_session(run) -> None:
    s, mm = run
    mm2 = MarketMaker(FixedSpread(half_spread=1))
    s2 = simulate(SHORT, 21, participant=mm2)
    assert s2.digest == s.digest
    assert mm2.inventory == mm.inventory and mm2.cash == mm.cash


def test_fixed_spread_quotes_and_inventory_limit() -> None:
    def view(mid, inv=0):
        return MarketView(0, 1, None, None, mid, inv, 100)

    fs = FixedSpread(half_spread=1, size=100, max_inventory=300)
    assert fs.quote(view(1_000_050)) == Quotes(999_900, 1_000_200, 100, 100)  # away from mid
    assert fs.quote(view(1_000_000)) == Quotes(999_900, 1_000_100, 100, 100)
    assert fs.quote(view(1_000_000, inv=300)).bid is None  # buying more would breach the limit
    assert fs.quote(view(1_000_000, inv=-300)).ask is None
    assert fs.quote(view(None)) == Quotes(None, None)


class FakeSim:
    """Just enough of the market's simulator to drive MarketMaker._place directly."""

    PARTICIPANT = 4

    def __init__(self) -> None:
        self.book = ArrayOrderBook(tick_size=100)
        self.p = MarketParams()
        self.now = 1
        self.n = 0

    def call(self, agent, fn, *args):
        return fn(*args)

    def new_id(self, agent):
        self.n += 1
        return order_id(agent, self.n)


def test_order_management_keeps_queue_priority_when_it_can() -> None:
    sim = FakeSim()
    mm = MarketMaker(FixedSpread())
    other = order_id(1, 1)
    sim.book.submit(other, Side.Sell, 1_000_000, 100)  # someone else's ask
    mm._place(sim, Side.Buy, 999_900, 200)
    mine = mm._live[Side.Buy]
    behind = order_id(1, 2)
    sim.book.submit(behind, Side.Buy, 999_900, 100)
    mm._place(sim, Side.Buy, 999_900, 150)  # size cut: keeps its place
    assert sim.book.queue(Side.Buy, 999_900) == [mine, behind]
    assert sim.book.find(mine).qty == 150
    mm._place(sim, Side.Buy, 999_900, 300)  # bigger size at the same price: left alone
    assert sim.book.find(mine).qty == 150
    mm._place(sim, Side.Buy, 999_800, 100)  # new price: cancel-replace
    assert sim.book.find(mine).price == 999_800
    mm._place(sim, Side.Buy, 1_000_100, 100)  # would cross the ask: clamped a tick below it
    assert sim.book.find(mine).price == 999_900
    mm._place(sim, Side.Buy, None, 100)
    assert sim.book.find(mine) is None and mm._live[Side.Buy] is None


def test_a_strategy_that_never_quotes_never_trades() -> None:
    class Silent:
        def quote(self, view):
            return Quotes(None, None)

    mm = MarketMaker(Silent())
    plain = simulate(SHORT, 3)
    s = simulate(SHORT, 3, participant=mm)
    assert mm.fills == [] and mm.orders_sent == 0
    assert s.digest == plain.digest


def test_inventory_half_life_of_an_ar1_path() -> None:
    mm = MarketMaker(FixedSpread(), requote_ms=1000)
    mm.samples = [(k * NS, int(round(10_000 * 0.9**k)), 0) for k in range(60)]
    s = simulate(MarketParams(hours=0.02), 1)
    stats = inventory_stats(s, mm)
    assert stats["half_life_s"] == pytest.approx(np.log(0.5) / np.log(0.9), rel=0.02)
    assert stats["max_abs_inventory"] == 10_000
