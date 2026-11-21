"""The risk overlay ported from the FYP (Project 3 step 6)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from mm_strategy.maker import MarketView, Quotes
from mm_strategy.market import MarketParams, simulate
from mm_strategy.risk import (
    REGIMES,
    AnomalyDetector,
    Policy,
    RegimeDetector,
    RegimeModel,
    RiskOverlay,
    RollingStats,
    kmeans,
)

TRACES = json.loads((Path(__file__).parent / "data" / "fyp_traces.json").read_text())
LEVELS = ("none", "moderate", "severe")
ACTIONS = ("allow", "throttle", "pause")


def test_policy_reproduces_the_fyp_decision_policy() -> None:
    pol = Policy()
    got = [ACTIONS.index(pol.decide(REGIMES[r], LEVELS[a]).value)
           for r, a in TRACES["policy"]["inputs"]]
    assert got == TRACES["policy"]["actions"]
    assert set(got) == {0, 1, 2}  # every action, so every rule, was exercised


def test_regime_gate_reproduces_the_fyp_cusum_and_persistence() -> None:
    det = RegimeDetector(model=None)
    got = [REGIMES.index(det.confirm(REGIMES[p], ret, sig))
           for p, ret, sig in TRACES["regime"]["inputs"]]
    assert got == TRACES["regime"]["labels"]


def test_anomaly_levels_reproduce_the_fyp_scorer() -> None:
    det = AnomalyDetector()
    got = [LEVELS.index(det.level(zr, zg, REGIMES[r])) for zr, zg, r in TRACES["anomaly"]["inputs"]]
    assert got == TRACES["anomaly"]["levels"]


def test_rolling_stats_match_numpy() -> None:
    x = np.random.default_rng(3).normal(5, 2, 500)
    rs = RollingStats(30)
    for i, v in enumerate(x):
        rs.update(v)
        w = x[max(0, i - 29): i + 1]
        assert rs.mean == pytest.approx(w.mean())
        if len(w) > 1:
            assert rs.std == pytest.approx(w.std(), rel=1e-9)  # population, as in the FYP


def test_kmeans_and_the_fyp_naming_of_clusters() -> None:
    rng = np.random.default_rng(0)
    # Centres in [vol_20, range_ratio, slope]: the FYP names them by volatility, then slope,
    # then range ratio.
    centres = {"calm": (1, 1, 0), "trend_up": (2, 1, 1), "trend_down": (2.1, 1, -1),
               "volatile": (5, 1, 0), "stressed": (5.1, 3, 0)}
    x = np.vstack([rng.normal(c, 0.05, (200, 3)) for c in centres.values()])
    c = kmeans(x, 5)
    assert sorted(np.round(c[:, 0], 0).tolist()) == [1, 2, 2, 5, 5]
    model = RegimeModel.fit(x)
    for name, centre in centres.items():
        assert model.label(np.array(centre, float)) == name


class Fixed:
    """A strategy that always quotes the same."""

    def quote(self, v):
        return Quotes(999_900, 1_000_100, 100, 100)


def view(inv=0, cash=0, mid=1_000_000.0, now=0):
    return MarketView(now, 3600 * 10**9, 999_900, 1_000_100, mid, inv, 100, (), (), cash)


def test_actions_edit_the_quotes() -> None:
    o = RiskOverlay(Fixed(), regimes=None, inventory_cap=None)
    assert o.quote(view()) == Quotes(999_900, 1_000_100, 100, 100)  # allow
    o.policy.action = o.policy.action.__class__("throttle")
    assert o.quote(view()) == Quotes(999_900, 1_000_100, 33, 33)
    o.policy.action = o.policy.action.__class__("pause")
    assert o.quote(view(inv=0)) == Quotes(None, None, 100, 100)  # flat: nothing to reduce
    assert o.quote(view(inv=60)) == Quotes(None, 1_000_100, 100, 60)  # sell, at most 60
    assert o.quote(view(inv=-250)) == Quotes(999_900, None, 100, 100)


def test_inventory_cap_and_kill_switch() -> None:
    capped = RiskOverlay(Fixed(), regimes=None, inventory_cap=500)
    assert capped.quote(view(inv=499)) == Quotes(999_900, 1_000_100, 100, 100)
    assert capped.quote(view(inv=500)) == Quotes(None, 1_000_100, 100, 100)
    k = RiskOverlay(Fixed(), regimes=None, inventory_cap=None, max_drawdown=1_000_000)
    k.quote(view(inv=0, cash=2_000_000, now=1))  # peak PnL 200 dollars
    assert k.killed_at is None
    q = k.quote(view(inv=100, cash=-99_000_000, now=2))  # marked: -99m + 100m = 1m: down $100
    assert k.killed_at == 2 and q == Quotes(None, 1_000_100, 100, 100)
    assert k.quote(view(inv=0, cash=5_000_000, now=3)) == Quotes(None, None, 100, 100)  # for good


def test_stress_episode_changes_only_its_window() -> None:
    p = MarketParams(hours=0.05)
    assert simulate(p, 4).digest == simulate(dataclasses.replace(p, stress_len_s=0.0,
                                                                 stress_vol=3.0), 4).digest
    s = simulate(dataclasses.replace(p, stress_start_s=60, stress_len_s=60, stress_vol=4.0), 4)
    r = np.diff(np.log(s.fair_value))
    t = np.arange(len(r)) * p.fv_step_ms / 1000
    inside = (t >= 60) & (t < 120)
    assert 3.5 < r[inside].std() / r[~inside].std() < 4.5
