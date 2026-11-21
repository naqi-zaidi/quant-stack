"""The synthetic market of Project 3 step 1."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mm_strategy.market import (
    INFORMED,
    NOISE_LIMIT,
    NOISE_MARKET,
    NS,
    MarketParams,
    agent_of,
    fair_value_path,
    simulate,
)
from mm_strategy.markouts import by_role, fill_markouts

SHORT = MarketParams(hours=0.05)  # three minutes


@pytest.fixture(scope="module")
def session():
    return simulate(SHORT, 11)


def test_same_seed_same_session_other_seed_differs(session) -> None:
    again = simulate(SHORT, 11)
    assert again.digest == session.digest
    for k in session.trades:
        assert np.array_equal(again.trades[k], session.trades[k])
    assert simulate(SHORT, 12).digest != session.digest


def test_each_agent_has_its_own_random_stream() -> None:
    # More informed traders change the book, but not the fair value, which has its own stream.
    a = simulate(SHORT, 5)
    b = simulate(replace(SHORT, informed_rate=2.0), 5)
    assert np.array_equal(a.fair_value, b.fair_value)
    assert a.digest != b.digest


def test_who_trades_with_whom(session) -> None:
    taker = agent_of(session.trades["taker"].astype(np.int64))
    maker = agent_of(session.trades["maker"].astype(np.int64))
    assert set(np.unique(maker)) == {NOISE_LIMIT}  # only resting noise orders provide liquidity
    assert set(np.unique(taker)) == {NOISE_MARKET, INFORMED}  # noise limits are never marketable
    assert session.calls["market maker"] == 0


def test_informed_traders_only_take_stale_quotes(session) -> None:
    tr = session.trades
    inf = agent_of(tr["taker"].astype(np.int64)) == INFORMED
    v = session.fair_value_at(tr["ts"][inf])
    gap = (v - tr["price"][inf]) * tr["taker_side"][inf]  # value captured per share
    assert gap.min() >= SHORT.edge_ticks * SHORT.tick - 1e-6


def test_fair_value_is_project_1_gbm() -> None:
    p = MarketParams(hours=2.0, sigma=0.4)
    fv = fair_value_path(p, np.random.default_rng(3))
    r = np.diff(np.log(fv))
    dt_years = p.fv_step_ms / 1000 / 3600 / (252 * 6.5)
    assert fv[0] == pytest.approx(p.s0 * 10_000)
    assert r.std() == pytest.approx(p.sigma * np.sqrt(dt_years), rel=0.01)


def test_markout_signs_and_horizons(session) -> None:
    m = fill_markouts(session, (0.001, 10))
    end = SHORT.duration_ns
    late = session.trades["ts"].astype(np.int64) + 10 * NS > end
    assert np.isnan(m[10][late]).all() and np.isfinite(m[10][~late]).all()
    # Every taker here trades only against noise limit orders, so each maker markout is the
    # negative of its counterparty's.
    roles = by_role(session, (10,))
    makers = [(role.removeprefix("maker vs "), row) for (_, role), row in roles.items()
              if role.startswith("maker vs")]
    assert len(makers) == 2
    for taker, row in makers:
        assert row[10] == pytest.approx(-roles[(taker, "taker")][10])
    assert np.isnan(session.mid_at(-1))  # no quote before the first order


def test_participant_hook_is_driven_by_the_loop() -> None:
    class Ticker:
        def __init__(self) -> None:
            self.seen: list[int] = []

        def on_start(self, sim) -> None:
            sim.schedule(NS, sim.PARTICIPANT, "tick")

        def on_event(self, sim, payload) -> None:
            assert payload == "tick"
            self.seen.append(sim.now)
            sim.schedule(sim.now + NS, sim.PARTICIPANT, "tick")

    t = Ticker()
    plain = simulate(SHORT, 4)
    with_ticker = simulate(SHORT, 4, participant=t)
    assert t.seen == [k * NS for k in range(1, int(SHORT.hours * 3600) + 1)]
    assert with_ticker.digest == plain.digest  # a participant that does nothing changes nothing
