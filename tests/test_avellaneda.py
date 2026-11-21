"""Avellaneda–Stoikov and its calibration (Project 3 step 3)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mm_strategy.calibration import (
    estimate_intensity,
    estimate_sigma,
    reach_per_order,
    true_sigma,
)
from mm_strategy.maker import AvellanedaStoikov, MarketMaker, MarketView, Quotes
from mm_strategy.market import NS, MarketParams, simulate

AS = AvellanedaStoikov(gamma=3e-4, sigma=1.4, k=0.5, size=100, max_inventory=2_000)


def view(mid, inv=0, now=0, end=3600 * NS):
    return MarketView(now, end, None, None, mid, inv, 100)


def test_formulas_by_hand() -> None:
    tau = 1800.0
    assert AS.reservation_offset(3, tau) == pytest.approx(-3 * 3e-4 * 1.4**2 * tau)
    assert AS.total_spread(tau) == pytest.approx(
        3e-4 * 1.4**2 * tau + 2 / 3e-4 * math.log(1 + 3e-4 / 0.5))
    # At the close only the intensity term is left; with no risk aversion it is 2/k.
    assert AS.total_spread(0) == pytest.approx(2 / 3e-4 * math.log1p(3e-4 / 0.5))
    no_skew = AvellanedaStoikov(0.0, 1.4, 0.5)
    assert no_skew.total_spread(1800) == pytest.approx(2 / 0.5)
    assert no_skew.reservation_offset(10, 1800) == 0
    tiny = AvellanedaStoikov(1e-9, 1.4, 0.5)
    assert tiny.total_spread(0) == pytest.approx(2 / 0.5, rel=1e-6)


def test_quotes_lean_against_inventory_and_round_away() -> None:
    flat = AS.quote(view(1_000_000))
    long = AS.quote(view(1_000_000, inv=500))
    short = AS.quote(view(1_000_000, inv=-500))
    assert long.bid < flat.bid and long.ask < flat.ask  # long: buy less, sell more
    assert short.bid > flat.bid and short.ask > flat.ask
    # Symmetric about r when flat, rounded away from it to whole ticks.
    half = AS.total_spread(3600) / 2
    assert flat == Quotes(math.floor(10_000 - half) * 100, math.ceil(10_000 + half) * 100)
    # The spread narrows into the close.
    late = AS.quote(view(1_000_000, now=3599 * NS))
    assert late.ask - late.bid < flat.ask - flat.bid
    assert AS.quote(view(1_000_000, inv=2_000)).bid is None  # hard limit still applies
    assert AS.quote(view(None)) == Quotes(None, None)


def test_true_sigma_in_ticks() -> None:
    p = MarketParams(sigma=0.3, s0=100.0)
    per_second = 0.3 / math.sqrt(252 * 6.5 * 3600)
    assert true_sigma(p) == pytest.approx(per_second * 100 * 100)  # $100 in cents


def test_reach_groups_the_trades_of_one_order() -> None:
    # Order 7 buys through three ask levels from a mid of 100 ticks; order 8 sells one tick down.
    trades = {
        "ts": np.array([10, 10, 10, 20]),
        "price": np.array([10_100, 10_200, 10_300, 9_900]),
        "taker": np.array([7, 7, 7, 8], dtype=np.uint64),
        "taker_side": np.array([1, 1, 1, -1], dtype=np.int8),
    }
    s = SimpleNamespace(trades=trades, params=SimpleNamespace(tick=100),
                        mid_at=lambda ts: np.full(len(np.atleast_1d(ts)), 10_000.0))
    assert reach_per_order(s).tolist() == [3.0, 1.0]


@pytest.fixture(scope="module")
def short_sessions():
    p = MarketParams(hours=0.25)
    return [simulate(p, 2000 + i) for i in range(4)]


def test_calibration_on_the_market(short_sessions) -> None:
    sig = estimate_sigma(short_sessions, 60.0)
    assert abs(sig.value - true_sigma(short_sessions[0].params)) < 5 * sig.se + 0.2
    fast = estimate_sigma(short_sessions, 1.0)
    assert fast.value > sig.value  # the bid-ask bounce inflates short intervals
    fit = estimate_intensity(short_sessions)
    assert fit.k.value > 0 and fit.A.value > 0 and fit.r2 > 0.9
    assert np.all(np.diff(fit.curve) <= 0)  # fewer orders reach further


def test_as_maker_runs_and_controls_inventory() -> None:
    p = MarketParams(hours=0.1)
    mm = MarketMaker(AvellanedaStoikov(1e-3, 1.36, 0.48))
    simulate(p, 5, participant=mm)
    q = np.array([x[1] for x in mm.samples])
    assert len(mm.fills) > 0 and np.abs(q).max() < 1_000
