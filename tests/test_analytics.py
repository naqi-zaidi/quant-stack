"""Microstructure analytics through the Python module (Project 2 step 6)."""

from __future__ import annotations

import doctest
import gc
from pathlib import Path

import numpy as np
import pytest

import orderbook.analytics as an
import orderbook.microstructure as ms
from orderbook import ArrayOrderBook, OrderBook, OrderType, Side
from orderbook import lobster as lb
from orderbook.analytics import (
    DepthProfile,
    OrderFlowImbalance,
    QueuePosition,
    QueuePositions,
    QueueStatus,
    SpreadDistribution,
)
from orderbook.gate import REGIMES

B, S = Side.Buy, Side.Sell


def all_four():
    return SpreadDistribution(1), DepthProfile(1, 4), OrderFlowImbalance(3, record=True), (
        QueuePositions()
    )


@pytest.mark.parametrize("book_cls", [OrderBook, ArrayOrderBook])
def test_hand_example(book_cls) -> None:
    spread, depth, ofi, queue = all_four()
    book = book_cls()
    for listener in (spread, depth, ofi, queue):
        book.attach(listener)
    book.submit(1, B, 99, 10, ts=10)  # bid appears: e = +10
    book.submit(2, S, 102, 5, ts=20)  # ask appears: e = -5; spread 3
    book.submit(3, S, 102, 7, ts=20)  # same ts, ask queue grows: e = -7
    book.submit(4, B, 97, 4, ts=30)  # 2 ticks behind the best bid: no quote update
    queue.track(3)
    book.submit(5, B, 102, 6, OrderType.IOC, ts=40)  # fills #2 (5) and 1 of #3: e = +5 +1

    assert spread.histogram(now=50) == [(3, 30)]
    assert spread.one_sided_ns(50) == 10 and spread.elapsed(50) == 40
    assert spread.quote == an.Quote((99, 10, 102, 6))
    assert depth.current(B) == [10, 0, 4, 0]
    assert depth.sums(B, 50) == [10 * 40, 0, 4 * 20, 0]
    assert depth.observed_ns(S, 50) == 30
    assert depth.mean(S, 50)[0] == pytest.approx((12 * 20 + 6 * 10) / 30)
    assert (ofi.total, ofi.updates, ofi.window_sum, ofi.last) == (10 - 5 - 7 + 6, 4, -5 - 7 + 6, 6)
    assert ofi.imbalance == pytest.approx((10 - 6) / 16)
    rec = ofi.records()
    assert rec["e"].tolist() == [10, -5, -7, 6]
    assert rec["ts"].dtype == np.uint64 and rec["e"].dtype == np.int64
    assert rec["ask"][0] == an.NO_ASK and rec["bid"][1] == 99
    assert queue.ahead(3) == QueuePosition((0, 0))
    assert queue.ahead(2) is None
    assert queue.orders(S) == [(3, 102, 6)]
    h = queue.history()
    assert h["id"].tolist() == [3, 3]
    assert h["qty_ahead"].tolist() == [5, 0] and h["leaves"].tolist() == [7, 6]
    assert h["status"].tolist() == [QueueStatus.Resting] * 2


def test_attach_rules_and_lifetimes() -> None:
    book = ArrayOrderBook()
    ofi = OrderFlowImbalance()
    book.attach(ofi)
    book.submit(1, B, 99, 10)
    with pytest.raises(RuntimeError, match="before the first event"):
        book.attach(SpreadDistribution())
    with pytest.raises(TypeError):
        book.attach(object())

    # The book keeps an attached listener alive.
    book2 = OrderBook()
    book2.attach(QueuePositions())
    gc.collect()
    book2.submit(1, S, 101, 5)  # would crash if the listener had been freed

    # Resetting keeps the rebuilt book.
    spread = SpreadDistribution()
    book3 = OrderBook()
    book3.attach(spread)
    book3.submit(1, B, 99, 1, ts=0)
    book3.submit(2, S, 100, 1, ts=0)
    spread.reset(100)
    assert spread.histogram(130) == [(1, 30)]


def test_queue_tracking_of_an_order_not_yet_sent() -> None:
    queue = QueuePositions()
    book = ArrayOrderBook()
    book.attach(queue)
    book.submit(1, S, 101, 10, ts=1)
    queue.track(2)  # not sent yet
    book.submit(2, S, 101, 5, ts=2)
    book.submit(3, B, 101, 15, OrderType.IOC, ts=3)
    h = queue.history()
    rows = zip(h["ts"].tolist(), h["qty_ahead"].tolist(), h["status"].tolist(), strict=True)
    assert list(rows) == [(2, 10, QueueStatus.Resting), (3, 0, QueueStatus.Filled)]


@pytest.mark.parametrize("book_cls", [OrderBook, ArrayOrderBook], ids=["map", "array"])
@pytest.mark.parametrize("regime", REGIMES[:3], ids=lambda r: r.name)
def test_synthetic_flow_matches_brute_force(regime, book_cls) -> None:
    rep, _ = ms.run_synthetic(regime, seed=7, calls=600, book_cls=book_cls)
    assert rep.problems == []
    assert rep.quote_updates > 0 and rep.positions > 0


def test_ofi_formula_matches_cpp_on_every_case() -> None:
    base = (100, 50, 102, 40)
    cases = [
        ((101, 10, 102, 40), 10),
        ((100, 70, 102, 40), 20),
        ((99, 30, 102, 40), -50),
        ((100, 50, 101, 15), -15),
        ((100, 50, 102, 25), 15),
        ((100, 50, 103, 60), 40),
        ((an.NO_BID, 0, 102, 40), -50),
        ((100, 50, an.NO_ASK, 0), 40),
    ]
    for cur, want in cases:
        assert ms.ofi_e(base, cur) == want
    assert ms.ofi_e(ms.EMPTY_QUOTE, (100, 50, an.NO_ASK, 0)) == 50


def test_replay_from_files_with_analytics(lobster_day: Path) -> None:
    msg = lobster_day
    rp = lb.ArrayReplay(2, 100)
    spread, ofi = SpreadDistribution(100), OrderFlowImbalance(record=True)
    rp.attach(spread)
    rp.attach(ofi)
    stats = rp.run(str(msg))
    assert stats.mismatches == 0
    # After the seed (t = 34200.001 s): bid only until the ask arrives at .002, then 2 cents
    # (5853500 - 5853300) for the rest of the day, to the last event at .005.
    ns = 1_000_000
    assert spread.one_sided_ns(34_200_005 * ns) == ns
    assert spread.histogram(34_200_005 * ns) == [(2, 3 * ns)]
    # The visible execution takes 10 of the 50 at the ask: +10.
    assert ofi.records()["e"].tolist() == [100, -50, 10]


@pytest.mark.parametrize("module", [an, ms])
def test_docstrings(module) -> None:
    assert doctest.testmod(module).failed == 0
