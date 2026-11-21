"""Discrete delta hedging: accounting, exact identities, and the second-order decomposition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mc_engine import bs_delta, bs_gamma, bs_price
from mc_engine.black_scholes import bs_vega
from mc_engine.hedging import delta_hedge, derman_kamal_sd, gbm_grid

S0, K, T, R, SI = 100.0, 100.0, 0.25, 0.03, 0.20


def reference_pnl(path, k, t, r, sig_i, sig_h, n, cost):
    """Scalar, step-by-step bookkeeping for one path: an independent check on the vector code."""
    step = (len(path) - 1) // n
    s = [float(x) for x in path[::step]]
    dt = t / n
    cash, pos = float(bs_price(s[0], k, t, r, sig_i)), 0.0
    for j in range(n):
        d = float(bs_delta(s[j], k, t - j * dt, r, sig_h))
        trade = d - pos
        cash -= trade * s[j] + cost * abs(trade) * s[j]
        cash *= math.exp(r * dt)
        pos = d
    delivered = 1.0 if s[-1] > k else 0.0
    cash -= cost * abs(pos - delivered) * s[-1]
    return math.exp(-r * t) * (cash + pos * s[-1] - max(s[-1] - k, 0.0))


def test_bs_gamma_is_the_derivative_of_delta():
    s, h = np.array([80.0, 100.0, 125.0]), 1e-4
    fd = (bs_delta(s + h, K, T, R, SI) - bs_delta(s - h, K, T, R, SI)) / (2 * h)
    np.testing.assert_allclose(bs_gamma(s, K, T, R, SI), fd, rtol=1e-6)
    fd_put = (bs_delta(s + h, K, T, R, SI, kind="put") - bs_delta(s - h, K, T, R, SI, kind="put"))
    np.testing.assert_allclose(bs_gamma(s, K, T, R, SI), fd_put / (2 * h), rtol=1e-6)


def test_vector_bookkeeping_matches_scalar_reference_with_costs():
    paths = gbm_grid(S0, 0.07, 0.3, T, 16, 5, seed=3)
    cost = 7e-4
    for sig_h, n in ((SI, 4), (0.3, 16), (0.12, 8)):
        h = delta_hedge(paths, K, T, R, SI, sig_h, n)
        ref = [reference_pnl(p, K, T, R, SI, sig_h, n, cost) for p in paths]
        np.testing.assert_allclose(h.net(cost), ref, rtol=0, atol=1e-12)


def test_one_path_set_serves_every_frequency():
    fine = gbm_grid(S0, R, 0.25, T, 64, 50, seed=1)
    a = delta_hedge(fine, K, T, R, SI, 0.25, 8)
    b = delta_hedge(fine[:, ::8], K, T, R, SI, 0.25, 8)
    for x, y in ((a.pnl, b.pnl), (a.gamma_pnl, b.gamma_pnl), (a.traded, b.traded)):
        np.testing.assert_array_equal(x, y)


def test_hedged_call_and_put_have_identical_pnl():
    """Put–call parity: the call hedge is the put hedge plus one share held throughout."""
    paths = gbm_grid(S0, 0.05, 0.25, T, 32, 400, seed=9)
    c = delta_hedge(paths, K, T, R, SI, 0.25, 32)
    p = delta_hedge(paths, K, T, R, SI, 0.25, 32, kind="put")
    np.testing.assert_allclose(c.pnl, p.pnl, atol=1e-10)
    np.testing.assert_allclose(c.gamma_pnl, p.gamma_pnl, atol=1e-12)
    np.testing.assert_allclose(c.traded - c.opening, p.traded - p.opening, atol=1e-9)
    assert c.locked_in == pytest.approx(p.locked_in, abs=1e-12)


def test_locked_in_premium_gap_and_opening_trade():
    paths = gbm_grid(S0, R, 0.15, T, 8, 10, seed=2)
    implied = delta_hedge(paths, K, T, R, SI, SI, 8)
    realized = delta_hedge(paths, K, T, R, SI, 0.15, 8)
    assert implied.locked_in == 0.0
    assert realized.locked_in == pytest.approx(
        float(bs_price(S0, K, T, R, SI) - bs_price(S0, K, T, R, 0.15)), abs=1e-12)
    assert realized.opening == pytest.approx(float(bs_delta(S0, K, T, R, 0.15)) * S0, abs=1e-12)
    assert np.all(realized.rebalance_traded >= 0)


@pytest.mark.parametrize("sig_r", [0.15, 0.25])
@pytest.mark.parametrize("hedge_at", ["implied", "realized"])
def test_martingale_identity_mean_pnl(sig_r, hedge_at):
    """Drift r: the mean P&L is BS(sigma_i) − BS(sigma_r) whatever the hedge."""
    paths = gbm_grid(S0, R, sig_r, T, 16, 20_000, seed=11)
    h = delta_hedge(paths, K, T, R, SI, SI if hedge_at == "implied" else sig_r, 16)
    m, se = h.mean_se()
    target = float(bs_price(S0, K, T, R, SI) - bs_price(S0, K, T, R, sig_r))
    assert abs(m - target) < 4 * se


def test_realized_vol_hedge_error_matches_derman_kamal():
    paths = gbm_grid(S0, R, SI, T, 256, 8_000, seed=5)
    for n in (64, 256):
        sd = delta_hedge(paths, K, T, R, SI, SI, n).pnl.std(ddof=1)
        assert sd / derman_kamal_sd(S0, K, T, R, SI, n) == pytest.approx(1.0, abs=0.06)


def test_implied_vol_hedge_tracks_the_gamma_term():
    paths = gbm_grid(S0, R, 0.15, T, 512, 4_000, seed=6)
    coarse = delta_hedge(paths, K, T, R, SI, SI, 32)
    fine = delta_hedge(paths, K, T, R, SI, SI, 512)
    for h in (coarse, fine):
        assert 1 - np.var(h.residual) / np.var(h.pnl) > 0.9
    lim = (SI**2 - 0.15**2) * fine.dollar_gamma
    assert np.corrcoef(fine.pnl, lim)[0, 1] > 0.95
    assert np.std(fine.pnl - lim) < 0.3 * np.std(coarse.pnl - lim)


def test_derman_kamal_formula():
    sd = derman_kamal_sd(S0, K, T, R, SI, 100)
    assert sd == pytest.approx(math.sqrt(math.pi / 400) * float(bs_vega(S0, K, T, R, SI)) * SI)
    assert derman_kamal_sd(S0, K, T, R, SI, 400) == pytest.approx(sd / 2)


def test_gbm_grid_reuses_normals_across_vols():
    a, b = gbm_grid(S0, R, 0.1, T, 10, 20, seed=4), gbm_grid(S0, R, 0.3, T, 10, 20, seed=4)
    dt = T / 10
    za = (np.diff(np.log(a), axis=1) - (R - 0.005) * dt) / (0.1 * np.sqrt(dt))
    zb = (np.diff(np.log(b), axis=1) - (R - 0.045) * dt) / (0.3 * np.sqrt(dt))
    np.testing.assert_allclose(za, zb, atol=1e-9)
    assert np.all(a[:, 0] == S0)


def test_rejects_bad_inputs():
    paths = gbm_grid(S0, R, SI, T, 12, 3, seed=0)
    with pytest.raises(ValueError, match="divide"):
        delta_hedge(paths, K, T, R, SI, SI, 5)
    with pytest.raises(ValueError, match="positive"):
        delta_hedge(-paths, K, T, R, SI, SI, 4)
    with pytest.raises(ValueError, match="shape"):
        delta_hedge(paths[0], K, T, R, SI, SI, 4)
    with pytest.raises(ValueError, match="positive"):
        delta_hedge(paths, K, T, R, 0.0, SI, 4)
