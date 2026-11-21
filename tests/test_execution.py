"""Almgren–Chriss schedules, the executor on the synthetic market, and impact calibration."""

from __future__ import annotations

import numpy as np
import pytest

from mm_strategy import execution as ex
from mm_strategy.market import EXECUTOR, NS, MarketParams, agent_of, simulate

X, N, TAU, SIG = 30_000.0, 60, 30.0, 1.3
MODEL = ex.ImpactModel(epsilon=0.6, eta=0.0025)


@pytest.mark.parametrize("lam", [1e-9, 1e-7, 1e-5, 1e-3])
def test_closed_form_is_the_minimiser_and_frontier_matches_sums(lam):
    closed = ex.ac_holdings(X, N, TAU, SIG, MODEL.eta, lam)
    numeric = ex.optimal_holdings(X, np.full(N, MODEL.eta), SIG, TAU, lam)
    np.testing.assert_allclose(closed, numeric, rtol=0, atol=1e-6)
    e, v = ex.frontier_point(X, N, TAU, SIG, MODEL, lam)
    assert e == pytest.approx(ex.expected_cost(closed, MODEL), rel=1e-12)
    assert v == pytest.approx(ex.variance(closed, SIG, TAU), rel=1e-12)


def test_minimiser_beats_perturbations():
    lam = 1e-6
    h = ex.ac_holdings(X, N, TAU, SIG, MODEL.eta, lam)
    best = ex.expected_cost(h, MODEL) + lam * ex.variance(h, SIG, TAU)
    rng = np.random.default_rng(0)
    for _ in range(20):
        g = h.copy()
        g[1:-1] += rng.normal(0, 50, N - 1)
        assert ex.expected_cost(g, MODEL) + lam * ex.variance(g, SIG, TAU) > best


def test_zero_risk_aversion_is_twap_and_fast_schedules_stay_finite():
    np.testing.assert_allclose(ex.ac_holdings(X, N, TAU, SIG, MODEL.eta, 0.0),
                               X * (1 - np.arange(N + 1) / N), atol=1e-9)
    e, v = ex.frontier_point(X, N, TAU, SIG, MODEL, 0.0)
    twap = ex.twap(int(X), N)
    assert e == pytest.approx(ex.expected_cost(twap, MODEL), rel=1e-12)
    assert v == pytest.approx(ex.variance(twap, SIG, TAU), rel=1e-12)
    h = ex.ac_holdings(X, N, TAU, SIG, MODEL.eta, 10.0)  # κT in the hundreds
    assert np.all(np.isfinite(h)) and h[0] == X and h[-1] == 0
    assert np.all(np.diff(h) <= 0)


def test_kappa_and_lambda_are_inverses():
    for kt in (0.5, 4.0, 40.0):
        lam = ex.lam_for_kappa(kt / (N * TAU), SIG, TAU, MODEL.eta)
        assert ex.kappa(lam, SIG, TAU, MODEL.eta) * N * TAU == pytest.approx(kt, rel=1e-10)


def test_time_varying_impact_at_zero_risk_aversion_sells_inversely_to_eta():
    eta = np.linspace(1e-3, 4e-3, N)
    n = -np.diff(ex.optimal_holdings(X, eta, SIG, TAU, 0.0))
    np.testing.assert_allclose(n, X * (1 / eta) / np.sum(1 / eta), rtol=1e-9)


def test_child_sizes_round_the_cumulative_amount():
    h = ex.ac_holdings(X, N, TAU, SIG, MODEL.eta, 1e-6)
    n = ex.child_sizes(h)
    assert n.sum() == int(X) and np.all(n >= 0)
    assert np.max(np.abs(np.cumsum(n) - (X - h[1:]))) <= 0.5
    w = np.arange(1, N + 1, dtype=float)
    v = ex.vwap(int(X), w)
    np.testing.assert_allclose(-np.diff(v), X * w / w.sum())


def test_variance_counts_holdings_between_children():
    h = np.array([300.0, 200.0, 100.0, 0.0])
    assert ex.variance(h, 2.0, 5.0) == pytest.approx(4 * 5 * (200**2 + 100**2))


SHORT = MarketParams(hours=0.1, profile_amp=2.0)


def run_short(sizes, seed=3, **kw):
    e = ex.Executor(30.0, 20.0, np.asarray(sizes), **kw)
    return e, simulate(SHORT, seed, e)


def test_executor_sells_exactly_and_the_shortfall_split_adds_up():
    sizes = ex.child_sizes(ex.twap(3_000, 12))
    e, s = run_short(sizes)
    sf = ex.shortfall(s, e, 30.0)
    assert sf.filled == 3_000
    assert sf.total == pytest.approx(sf.arrival + sf.timing + sf.impact, abs=1e-9)
    trades = s.trades
    mine = agent_of(trades["taker"]) == EXECUTOR
    assert trades["qty"][mine].sum() == 3_000 and np.all(trades["taker_side"][mine] == -1)
    assert sf.impact > 0  # selling into the bid costs something on average


def test_unfilled_shares_roll_forward_and_are_retried():
    sizes = np.zeros(12, dtype=int)
    sizes[0] = 40_000  # far more than the bid side holds
    e, _ = run_short(sizes)
    filled = sum(c[2] for c in e.children)
    assert e.children[0][2] < 40_000
    assert len(e.children) > 1 and filled + e.left == 40_000


def test_idle_executor_leaves_the_session_unchanged():
    e, s = run_short(np.zeros(12, dtype=int))
    plain = simulate(SHORT, 3)
    assert s.digest == plain.digest and not e.fills
    assert np.array_equal(s.fair_value, plain.fair_value)


def test_probe_costs_sign_and_timing_variance():
    sides = np.where(np.arange(12) % 2 == 0, -1, 1)
    e, _ = run_short(np.full(12, 300), sides=sides, carry=False)
    t, n, c = ex.probe_costs(e)
    assert len(t) == len(n) == len(c) and np.all(n > 0)
    assert np.mean(c) > 0  # both buys and sells pay against the mid
    sells, _ = run_short(np.full(12, 100))
    ts = np.array([ch[0] for ch in sells.children]) / NS
    left = 1_200 - np.cumsum([ch[2] for ch in sells.children])
    assert sells.timing_variance(2.0) == pytest.approx(4 * np.sum(left[:-1] ** 2 * np.diff(ts)))


def test_impact_fits_recover_known_coefficients():
    rng = np.random.default_rng(1)
    n = rng.integers(50, 1500, 4_000).astype(float)
    t = rng.uniform(0, 600, 4_000)
    eta_b = np.array([1e-3, 3e-3, 2e-3])
    b = ex.bucket_of(t, 0, 600, 3)
    cost = 0.5 + eta_b[b] * n + rng.normal(0, 0.05, 4_000)
    fit = ex.fit_bucket_impact(t, n, cost, 0, 600, 3)
    assert fit.epsilon == pytest.approx(0.5, abs=0.01)
    np.testing.assert_allclose(fit.eta, eta_b, rtol=0.01)
    one = ex.fit_impact(n, 0.4 + 2e-3 * n)
    assert (one.epsilon, one.eta, one.r2) == pytest.approx((0.4, 2e-3, 1.0))


def test_activity_profile_is_u_shaped_and_volume_profile_sums_to_one():
    p = MarketParams(profile_amp=2.0)
    assert p.activity(0.0) == pytest.approx(3.0) and p.activity(1800.0) == pytest.approx(1.0)
    s = simulate(p, 11)
    prof = ex.volume_profile([s], 0.0, 3600.0, 6)
    assert prof.sum() == pytest.approx(1.0)
    assert prof[0] > prof[2] and prof[-1] > prof[3]
