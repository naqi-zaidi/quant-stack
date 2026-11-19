"""Merton jump-diffusion and implied volatility."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mc_engine import bs_price
from mc_engine.implied_vol import implied_vol
from mc_engine.merton import MertonParams, merton_price, merton_terminal, price_merton_mc

S0, T, R, Q = 100.0, 1.0, 0.05, 0.01
P = MertonParams(sigma=0.15, lam=0.5, mu_j=-0.10, delta=0.15)
Z_TOL = 4.0


def test_no_jumps_is_black_scholes():
    p = MertonParams(sigma=0.2, lam=0.0, mu_j=-0.1, delta=0.1)
    for kind in ("call", "put"):
        assert merton_price(S0, 95, T, R, p, Q, kind) == pytest.approx(
            float(bs_price(S0, 95, T, R, 0.2, Q, kind)), rel=1e-12
        )


def test_put_call_parity():
    c = merton_price(S0, 105, T, R, P, Q, "call")
    p = merton_price(S0, 105, T, R, P, Q, "put")
    assert c - p == pytest.approx(S0 * np.exp(-Q * T) - 105 * np.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("k", [80.0, 100.0, 120.0])
def test_exact_simulation_matches_series(kind, k):
    res = price_merton_mc(S0, k, T, R, P, Q, kind, 1 << 18, seed=[int(k), len(kind)])
    ref = merton_price(S0, k, T, R, P, Q, kind)
    assert abs(res.z_score(ref)) < Z_TOL, (res, ref)


def test_discounted_price_is_a_martingale():
    x = np.exp(-(R - Q) * T) * merton_terminal(S0, T, R, P, Q, np.random.default_rng(1), 1 << 18)
    assert abs(x.mean() - S0) < Z_TOL * x.std(ddof=1) / np.sqrt(x.size)


def test_negative_jumps_produce_a_downward_skew():
    vols = [implied_vol(merton_price(S0, k, T, R, P, Q), S0, k, T, R, Q) for k in (80, 100, 120)]
    assert vols[0] > vols[1]  # crash risk makes downside strikes dearer


@given(
    st.floats(0.05, 1.5),
    st.floats(0.6, 1.6),
    st.floats(0.05, 3.0),
    st.sampled_from(["call", "put"]),
)
def test_implied_vol_round_trips(sigma, moneyness, t, kind):
    k = S0 * moneyness
    price = float(bs_price(S0, k, t, R, sigma, Q, kind))
    vega_floor = 1e-8 * S0  # skip prices too flat in sigma to invert meaningfully
    if price - float(bs_price(S0, k, t, R, sigma * 0.99, Q, kind)) < vega_floor:
        return
    assert implied_vol(price, S0, k, t, R, Q, kind) == pytest.approx(sigma, rel=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(0.0, S0, 100, T, R, Q))
    assert np.isnan(implied_vol(S0, S0, 100, T, R, Q))  # a call can't be worth the stock
