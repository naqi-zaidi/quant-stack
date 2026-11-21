"""Order-flow signals and the walk-forward (Project 3 step 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mm_strategy.maker import AvellanedaStoikov, MarketMaker, MarketView
from mm_strategy.market import MarketParams, order_id, simulate
from mm_strategy.signals import (
    FEATURES,
    FlowFeatures,
    FlowModel,
    FlowSkewAS,
    fit,
    r2_oos,
    training_rows,
    walk_forward,
)
from orderbook import ArrayOrderBook, Side

TICK = 100
SHORT = MarketParams(hours=0.05)
AS = AvellanedaStoikov(gamma=3e-4, sigma=1.4, k=0.5)


def view(bids, asks, now=0, inv=0):
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    mid = (bid + ask) / 2 if bids and asks else None
    return MarketView(now, 3600 * 10**9, bid, ask, mid, inv, TICK, tuple(bids), tuple(asks))


class FakeSim:
    """Just enough of the simulator for MarketMaker._place (as in test_maker)."""

    def __init__(self) -> None:
        self.book = ArrayOrderBook(tick_size=TICK)
        self.p = MarketParams()
        self.now = 1
        self.n = 0

    def call(self, agent, fn, *args):
        return fn(*args)

    def new_id(self, agent):
        self.n += 1
        return order_id(agent, self.n)


def test_maker_view_takes_out_its_own_orders() -> None:
    sim = FakeSim()
    mm = MarketMaker(AS)
    book = sim.book
    book.submit(order_id(1, 1), Side.Buy, 999_900, 300)
    book.submit(order_id(1, 2), Side.Buy, 999_700, 200)
    book.submit(order_id(1, 3), Side.Sell, 1_000_200, 400)
    mm._place(sim, Side.Buy, 999_900, 100)  # joins someone else's level
    mm._place(sim, Side.Sell, 1_000_100, 100)  # a level of its own
    assert mm._market_side(book, Side.Buy) == ((999_900, 300), (999_700, 200))
    assert mm._market_side(book, Side.Sell) == ((1_000_200, 400),)


def test_features_by_hand() -> None:
    ff = FlowFeatures(step_s=1.0, ofi_window_s=2.0, depth_ticks=2, lag_s=1.0)
    # t0: no previous snapshot, so e = 0; windows not full.
    assert ff.update(view([(1000, 300), (900, 100)], [(1100, 200)], now=0)) is None
    # t1: bid unchanged with more size (+100), ask unchanged (no change): e = +100.
    x = ff.update(view([(1000, 400), (900, 100)], [(1100, 200), (1200, 500)], now=1))
    # OFI over the last 2 snapshots: 0 + 100 shares = 1 hundred. Depth within 2 ticks: bid
    # 400 + 100, ask 200 + 500. Mid unchanged.
    assert x == pytest.approx([1.0, (500 - 700) / 1200, 0.0])
    # t2: the ask is swept to 1200 (old ask queue of 200 gone: +200), bid steps up to 1100
    # with 50 (+50). e = 250; window now holds 100 and 250.
    x = ff.update(view([(1100, 50), (1000, 400)], [(1200, 500)], now=2))
    assert x == pytest.approx([3.5, (450 - 500) / 950, 1.0])  # mid 10.5 -> 11.5 ticks
    # An empty side gives no features, and resets the OFI chain.
    assert ff.update(view([(1100, 50)], [], now=3)) is None
    assert len(ff.log) == 4
    assert math.isnan(ff.log[0][2]) and math.isnan(ff.log[3][1])
    assert ff.log[2][1:] == pytest.approx((11.5, 3.5, (450 - 500) / 950, 1.0))


def test_fit_recovers_coefficients_and_scores_out_of_sample() -> None:
    rng = np.random.default_rng(0)
    beta = np.array([-0.05, 1.3, -0.16])

    def log(n):
        x = rng.normal(size=(n, 3))
        dmid = x @ beta + rng.normal(scale=0.5, size=n)
        mid = np.r_[0.0, np.cumsum(dmid)]
        return [(i, mid[i], *x[i]) for i in range(n)] + [(n, mid[n], *(np.nan,) * 3)]

    train = [log(4_000) for _ in range(3)]
    m = fit(train, horizon_s=1.0, step_s=1.0)
    assert m.coef == pytest.approx(beta, abs=0.03)
    assert (m.rows, m.sessions) == (12_000, 3)
    test = log(4_000)
    x, y = training_rows(test, 1.0, 1.0)
    signal = x @ beta
    expected = 1 - ((y - signal) ** 2).sum() / (y**2).sum()
    assert r2_oos(FlowModel(tuple(beta), 1.0, 0, 0), test, 1.0) == pytest.approx(expected)
    assert r2_oos(FlowModel((0.0, 0.0, 0.0), 1.0, 0, 0), test, 1.0) == 0.0
    # A model on one feature only.
    m1 = fit(train, 1.0, 1.0, columns=(1,))
    assert m1.columns == (1,) and m1.coef == pytest.approx([1.3], abs=0.05)
    assert m1.predict(np.array([9.0, 1.0, 9.0])) == pytest.approx(m1.coef[0])


def test_walk_forward_never_sees_the_day_it_trades_or_later() -> None:
    rng = np.random.default_rng(1)

    def random_log(n=500):
        mid = np.cumsum(rng.normal(size=n))
        return [(i, mid[i], *rng.normal(size=3)) for i in range(n)]

    history = [random_log() for _ in range(2)]
    future = [random_log() for _ in range(4)]
    seen = []

    def trade(day, model):
        seen.append((day, model))
        return day * 10, future[day]

    out = walk_forward(history, range(4), trade, horizon_s=0.1)
    assert [d for d, _ in seen] == [0, 1, 2, 3]
    for day, model, result in out:
        assert result == day * 10
        assert model.sessions == 2 + day
        assert model == fit(history + future[:day], 0.1)
    # Rewriting a later day's log leaves every earlier day's model alone.
    future[3] = random_log()
    out2 = walk_forward(history, range(4), lambda d, m: (None, future[d]), horizon_s=0.1)
    assert [m for _, m, _ in out2] == [m for _, m, _ in out]


def test_skew_moves_both_quotes_by_the_prediction() -> None:
    bids, asks = [(999_900, 300), (999_800, 300)], [(1_000_100, 300), (1_000_200, 300)]
    v = view(bids, asks)
    assert FlowSkewAS(AS).quote(v) == AS.quote(v)  # no model: plain A–S
    # A model whose prediction is exactly +2 ticks whatever the features.
    up2 = FlowSkewAS(AS, FlowModel((0.0, 2.0, 0.0), 5.0, 0, 0), features=FlowFeatures(
        step_s=1.0, ofi_window_s=1.0, lag_s=1.0))
    up2.features.update(view(bids, [(1_000_100, 1)]))  # fills the one-step windows
    v2 = view(bids, [(1_000_100, 1)], now=1)  # depth imbalance (600 - 1) / 601, about +1
    q = up2.quote(v2)
    x = up2.features.log[-1][2:]
    shift = 2.0 * x[1]
    plain = AS.quote_about(v2, 0.0)
    assert up2.shifts[-1] == pytest.approx(shift)
    assert q == AS.quote_about(v2, shift)
    assert q.bid >= plain.bid and q.ask >= plain.ask and q.bid > plain.bid


def test_a_model_free_flow_maker_trades_exactly_like_plain_as() -> None:
    plain = MarketMaker(AS)
    s1 = simulate(SHORT, 5, participant=plain)
    st = FlowSkewAS(AS)
    flow = MarketMaker(st)
    s2 = simulate(SHORT, 5, participant=flow)
    assert s2.digest == s1.digest
    assert (flow.inventory, flow.cash) == (plain.inventory, plain.cash)
    assert len(st.features.log) == len(flow.samples)
    x, y = training_rows(st.features.log, 5.0)
    assert len(y) > 1_000 and x.shape[1] == len(FEATURES)
