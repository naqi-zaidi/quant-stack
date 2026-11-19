"""Heston: the semi-analytic price is checked against published benchmarks and limits; the
simulation schemes are checked against the semi-analytic price."""

import numpy as np
import pytest

from mc_engine import bs_price
from mc_engine.heston import (
    HestonParams,
    char_func,
    effective_vol,
    heston_price,
    price_heston_mc,
    simulate_log_terminal,
)
from mc_engine.implied_vol import implied_vol

S0, T, R, Q = 100.0, 1.0, 0.05, 0.01
EASY = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
HARSH = HestonParams(kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, v0=0.04)  # Feller badly violated
Z_TOL = 4.0


def test_characteristic_function_is_normalised_and_a_martingale():
    assert char_func(0.0, S0, T, R, EASY, Q) == pytest.approx(1.0, abs=1e-14)
    assert char_func(-1j, S0, T, R, EASY, Q).real == pytest.approx(
        S0 * np.exp((R - Q) * T), rel=1e-13
    )


# "Group Quantization and Mellin Representations of the Heston Model",
# arXiv:2606.13992, Table 4: S0 = K = 100, r = 0.03, q = 0, T = 1, kappa = 2, theta = 0.04,
# xi = 0.35, rho = -0.7, v0 = 0.04, varying one parameter. Price rises with v0 and theta, so the
# range endpoints are the prices at the lowest and highest values tested. The paper's two
# methods differ by up to 1.5e-7, so that is the tolerance.
@pytest.mark.parametrize(
    ("param", "value", "benchmark"),
    [
        ("v0", 0.01, 7.85571767),
        ("v0", 0.09, 10.99728901),
        ("theta", 0.01, 7.26603514),
        ("theta", 0.09, 11.56109741),
    ],
)
def test_matches_published_benchmark(param, value, benchmark):
    base = {"kappa": 2.0, "theta": 0.04, "xi": 0.35, "rho": -0.7, "v0": 0.04}
    p = HestonParams(**(base | {param: value}))
    assert heston_price(100.0, 100.0, 1.0, 0.03, p) == pytest.approx(benchmark, abs=2e-7)


def test_vanishing_vol_of_vol_is_black_scholes_at_the_effective_vol():
    gaps = []
    for xi in (1e-2, 1e-3):
        p = HestonParams(2.0, 0.04, xi, -0.7, 0.09)
        bs = float(bs_price(S0, 100, T, R, effective_vol(T, p), Q))
        gaps.append(heston_price(S0, 100, T, R, p, Q) - bs)
    assert abs(gaps[1]) < 1e-3
    assert abs(gaps[1]) < abs(gaps[0]) / 5  # the gap is O(xi)


def test_put_call_parity():
    c = heston_price(S0, 110, T, R, EASY, Q, "call")
    p = heston_price(S0, 110, T, R, EASY, Q, "put")
    assert c - p == pytest.approx(S0 * np.exp(-Q * T) - 110 * np.exp(-R * T), abs=1e-9)


def test_negative_correlation_produces_a_downward_skew():
    vols = [implied_vol(heston_price(S0, k, T, R, EASY, Q), S0, k, T, R, Q) for k in (80, 100, 120)]
    assert vols[0] > vols[1] > vols[2]


@pytest.mark.parametrize("scheme", ["euler", "qe"])
@pytest.mark.parametrize("k", [80.0, 100.0, 120.0])
def test_fine_step_simulation_matches_semi_analytic(scheme, k):
    res = price_heston_mc(
        S0,
        k,
        T,
        R,
        EASY,
        Q,
        "call",
        n_steps=128,
        n_paths=1 << 17,
        seed=[int(k), len(scheme)],
        scheme=scheme,
    )
    ref = heston_price(S0, k, T, R, EASY, Q)
    assert abs(res.z_score(ref)) < Z_TOL, (scheme, k, res, ref)


def test_qe_is_far_less_biased_than_euler_at_coarse_steps_when_feller_fails():
    ref = heston_price(S0, 100, 5.0, 0.0, HARSH)
    euler = price_heston_mc(
        S0, 100, 5.0, 0.0, HARSH, n_steps=20, n_paths=1 << 16, seed=1, scheme="euler"
    )
    qe = price_heston_mc(S0, 100, 5.0, 0.0, HARSH, n_steps=20, n_paths=1 << 16, seed=1)
    assert euler.z_score(ref) > 10  # Euler: large, significant upward bias
    assert abs(qe.price - ref) < abs(euler.price - ref) / 10


def test_qe_paths_stay_finite_when_feller_fails():
    # Variance hits zero constantly here; QE's zero-mass branch must not produce NaN or inf.
    rng = np.random.default_rng(3)
    x = simulate_log_terminal(S0, 5.0, 0.0, HARSH, 0.0, 10, 1 << 14, rng, "qe")
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize("kwargs", [{"scheme": "milstein"}, {"n_steps": 0}, {"n_paths": 1}])
def test_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        price_heston_mc(S0, 100, T, R, EASY, Q, **kwargs)
