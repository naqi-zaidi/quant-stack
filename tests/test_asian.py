"""Asian options: the geometric closed form checks the path machinery exactly; the arithmetic
estimators must agree with each other, and the control variate must pay for itself."""

import numpy as np
import pytest

from mc_engine import bs_price
from mc_engine.asian import geometric_asian_price, price_asian

S0, R, Q, SIGMA, T = 100.0, 0.05, 0.01, 0.25, 1.0
Z_TOL = 4.0
N = 1 << 16


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("k", [90.0, 100.0, 110.0])
@pytest.mark.parametrize("sampler", ["pseudo", "sobol"])
def test_geometric_mc_matches_closed_form(kind, k, sampler):
    res = price_asian(
        S0,
        k,
        T,
        R,
        SIGMA,
        Q,
        kind,
        12,
        N,
        seed=[int(k), len(kind)],
        average="geometric",
        sampler=sampler,
    )
    ref = geometric_asian_price(S0, k, T, R, SIGMA, Q, kind, 12)
    assert abs(res.z_score(ref)) < Z_TOL, (res, ref)


def test_one_date_geometric_asian_is_european():
    # With a single monitoring date at T, G = S_T and the closed form must collapse to BS.
    for kind in ("call", "put"):
        assert geometric_asian_price(S0, 95, T, R, SIGMA, Q, kind, 1) == pytest.approx(
            float(bs_price(S0, 95, T, R, SIGMA, Q, kind)), rel=1e-12
        )


def test_geometric_put_call_parity():
    n = 12
    c = geometric_asian_price(S0, 100, T, R, SIGMA, Q, "call", n)
    p = geometric_asian_price(S0, 100, T, R, SIGMA, Q, "put", n)
    mu = np.log(S0) + (R - Q - 0.5 * SIGMA**2) * T * (n + 1) / (2 * n)
    var = SIGMA**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    assert c - p == pytest.approx(np.exp(-R * T) * (np.exp(mu + var / 2) - 100), rel=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_arithmetic_estimators_agree(kind):
    ref = price_asian(
        S0, 100, T, R, SIGMA, Q, kind, 12, 1 << 18, seed=1, control_variate=True, sampler="sobol"
    )
    for cv, sampler in [(False, "pseudo"), (True, "pseudo"), (False, "sobol")]:
        res = price_asian(
            S0, 100, T, R, SIGMA, Q, kind, 12, N, seed=2, control_variate=cv, sampler=sampler
        )
        z = (res.price - ref.price) / np.hypot(res.std_error, ref.std_error)
        assert abs(z) < Z_TOL, (cv, sampler, res, ref)


def test_arithmetic_call_is_above_geometric_call():
    # AM >= GM path by path, and the call payoff is increasing, so the prices are ordered.
    a = price_asian(S0, 100, T, R, SIGMA, Q, "call", 12, N, seed=3, control_variate=True)
    assert a.price > geometric_asian_price(S0, 100, T, R, SIGMA, Q, "call", 12)


def test_control_variate_cuts_variance_by_two_orders_of_magnitude():
    plain = price_asian(S0, 100, T, R, SIGMA, Q, "call", 12, N, seed=4)
    cv = price_asian(S0, 100, T, R, SIGMA, Q, "call", 12, N, seed=4, control_variate=True)
    assert (plain.std_error / cv.std_error) ** 2 > 100


def test_low_volatility_limit_is_the_discounted_forward_average():
    n = 12
    fwd = np.mean([S0 * np.exp((R - Q) * T * i / n) for i in range(1, n + 1)])
    res = price_asian(S0, 90, T, R, 1e-4, Q, "call", n, 1 << 12, seed=5)
    assert res.price == pytest.approx(np.exp(-R * T) * (fwd - 90), rel=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"average": "harmonic"},
        {"average": "geometric", "control_variate": True},
        {"kind": "digital"},
        {"sampler": "lattice"},
        {"n_steps": 0},
    ],
)
def test_rejects_bad_inputs(kwargs):
    args = {"s0": S0, "k": 100.0, "t": T, "r": R, "sigma": SIGMA, "n_paths": 1 << 10} | kwargs
    with pytest.raises(ValueError):
        price_asian(**args)
