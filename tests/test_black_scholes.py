"""Closed-form checks: textbook values, finite-difference Greeks, and no-arbitrage properties."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mc_engine import bs_delta, bs_price, bs_vega


@pytest.mark.parametrize(
    ("s", "k", "t", "r", "sigma", "call", "put"),
    [
        (100, 100, 1.0, 0.05, 0.2, 10.450583572185565, 5.573526022256971),
        (42, 40, 0.5, 0.10, 0.2, 4.759422392871532, 0.808599372900092),  # Hull, Example 15.6
    ],
)
def test_textbook_values(s, k, t, r, sigma, call, put):
    assert bs_price(s, k, t, r, sigma, kind="call") == pytest.approx(call, rel=1e-12)
    assert bs_price(s, k, t, r, sigma, kind="put") == pytest.approx(put, rel=1e-12)


def test_broadcasts_over_strikes():
    k = np.array([80.0, 100.0, 120.0])
    prices = bs_price(100, k, 1.0, 0.05, 0.2)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # calls get cheaper as the strike rises


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_finite_difference(kind):
    h = 1e-4
    up = bs_price(100 + h, 95, 0.75, 0.03, 0.25, q=0.01, kind=kind)
    dn = bs_price(100 - h, 95, 0.75, 0.03, 0.25, q=0.01, kind=kind)
    assert bs_delta(100, 95, 0.75, 0.03, 0.25, q=0.01, kind=kind) == pytest.approx(
        (up - dn) / (2 * h), rel=1e-7
    )


def test_vega_matches_finite_difference():
    h = 1e-5
    up = bs_price(100, 95, 0.75, 0.03, 0.25 + h, q=0.01)
    dn = bs_price(100, 95, 0.75, 0.03, 0.25 - h, q=0.01)
    assert bs_vega(100, 95, 0.75, 0.03, 0.25, q=0.01) == pytest.approx(
        (up - dn) / (2 * h), rel=1e-7
    )


def test_rejects_unknown_kind():
    with pytest.raises(ValueError):
        bs_price(100, 100, 1.0, 0.05, 0.2, kind="straddle")


# Market-plausible parameter ranges for the property tests.
spot = st.floats(1.0, 1_000.0)
moneyness = st.floats(0.5, 2.0)
maturity = st.floats(0.01, 5.0)
rate = st.floats(-0.02, 0.10)
dividend = st.floats(0.0, 0.05)
vol = st.floats(0.01, 1.5)


@given(spot, moneyness, maturity, rate, dividend, vol)
def test_put_call_parity(s, m, t, r, q, sigma):
    k = s * m
    lhs = bs_price(s, k, t, r, sigma, q, "call") - bs_price(s, k, t, r, sigma, q, "put")
    rhs = s * np.exp(-q * t) - k * np.exp(-r * t)
    assert lhs == pytest.approx(rhs, abs=1e-9 * max(s, k))


@given(spot, moneyness, maturity, rate, dividend, vol)
def test_call_within_no_arbitrage_bounds(s, m, t, r, q, sigma):
    k = s * m
    c = bs_price(s, k, t, r, sigma, q, "call")
    lower = max(s * np.exp(-q * t) - k * np.exp(-r * t), 0.0)
    upper = s * np.exp(-q * t)
    tol = 1e-9 * max(s, k)
    assert lower - tol <= c <= upper + tol


@given(spot, moneyness, maturity, rate, dividend, vol, st.floats(1e-4, 0.5))
def test_price_increases_with_volatility(s, m, t, r, q, sigma, bump):
    k = s * m
    lo = bs_price(s, k, t, r, sigma, q, "call")
    hi = bs_price(s, k, t, r, sigma + bump, q, "call")
    assert hi >= lo - 1e-9 * max(s, k)
