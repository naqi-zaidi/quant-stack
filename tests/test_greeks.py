"""Greek estimators against closed forms, plus the textbook failure mode of pathwise."""

import numpy as np
import pytest

from mc_engine import bs_delta, bs_vega
from mc_engine.black_scholes import (
    bs_digital_call,
    bs_digital_call_delta,
    bs_digital_call_vega,
)
from mc_engine.greeks import ESTIMATORS, mc_greeks

S0, R, Q, SIGMA, T = 100.0, 0.05, 0.01, 0.25, 1.0
Z_TOL = 4.0
N = 1 << 17


def _reference(kind, k):
    if kind == "digital_call":
        return (
            bs_digital_call_delta(S0, k, T, R, SIGMA, Q),
            bs_digital_call_vega(S0, k, T, R, SIGMA, Q),
        )
    return bs_delta(S0, k, T, R, SIGMA, Q, kind), bs_vega(S0, k, T, R, SIGMA, Q)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("k", [80.0, 100.0, 120.0])
def test_all_estimators_match_black_scholes(kind, k):
    res = mc_greeks(S0, k, T, R, SIGMA, Q, kind, N, seed=[int(k), len(kind)])
    ref_delta, ref_vega = _reference(kind, k)
    for est in ESTIMATORS:
        assert abs(res.delta[est].z_score(ref_delta)) < Z_TOL, ("delta", est, res.delta[est])
        assert abs(res.vega[est].z_score(ref_vega)) < Z_TOL, ("vega", est, res.vega[est])


def test_pathwise_fails_on_a_digital_while_likelihood_ratio_works():
    res = mc_greeks(S0, 100.0, T, R, SIGMA, Q, "digital_call", N, seed=3)
    ref_delta, ref_vega = _reference("digital_call", 100.0)
    # Pathwise is confidently wrong: exactly zero, with zero standard error.
    assert res.delta["pathwise"].value == 0.0
    assert res.delta["pathwise"].std_error == 0.0
    assert ref_delta > 0.01
    # Likelihood ratio and the CRN bump both recover the true sensitivities.
    for est in ("likelihood_ratio", "bump_crn"):
        assert abs(res.delta[est].z_score(ref_delta)) < Z_TOL
        assert abs(res.vega[est].z_score(ref_vega)) < Z_TOL


def test_pathwise_has_lower_variance_than_likelihood_ratio_for_a_call():
    res = mc_greeks(S0, 100.0, T, R, SIGMA, Q, "call", N, seed=4)
    assert res.delta["pathwise"].std_error < res.delta["likelihood_ratio"].std_error / 2
    assert res.vega["pathwise"].std_error < res.vega["likelihood_ratio"].std_error / 2


def _lr_over_pathwise_variance(k, t):
    d = mc_greeks(S0, k, t, R, SIGMA, Q, "call", N, seed=5).delta
    return (d["likelihood_ratio"].std_error / d["pathwise"].std_error) ** 2


def test_likelihood_ratio_variance_explodes_at_short_maturity_in_the_money():
    ratios = [_lr_over_pathwise_variance(80.0, t) for t in (2.0, 0.25, 0.05)]
    assert ratios[0] < ratios[1] < ratios[2]
    assert ratios[2] > 100 * ratios[0]


def test_likelihood_ratio_variance_is_flat_at_the_money():
    # The payoff shrinks like sqrt(T) too, cancelling the 1/sqrt(T) weight; the limit is 5.
    for t in (2.0, 0.25, 0.01):
        assert 4.0 < _lr_over_pathwise_variance(100.0, t) < 7.0


def test_call_put_delta_parity():
    # Per path, pathwise call delta - put delta = e^{-rT} S_T / S0, whose mean is e^{-qT}.
    c = mc_greeks(S0, 100.0, T, R, SIGMA, Q, "call", N, seed=6).delta["pathwise"]
    p = mc_greeks(S0, 100.0, T, R, SIGMA, Q, "put", N, seed=6).delta["pathwise"]
    se = np.hypot(c.std_error, p.std_error)
    assert abs((c.value - p.value) - np.exp(-Q * T)) < Z_TOL * se


def test_digital_closed_forms_match_finite_differences():
    h = 1e-4
    d = bs_digital_call(S0 + h, 95, T, R, SIGMA, Q) - bs_digital_call(S0 - h, 95, T, R, SIGMA, Q)
    v = bs_digital_call(S0, 95, T, R, SIGMA + h, Q) - bs_digital_call(S0, 95, T, R, SIGMA - h, Q)
    assert bs_digital_call_delta(S0, 95, T, R, SIGMA, Q) == pytest.approx(d / (2 * h), rel=1e-6)
    assert bs_digital_call_vega(S0, 95, T, R, SIGMA, Q) == pytest.approx(v / (2 * h), rel=1e-6)


def test_same_seed_reproduces():
    a = mc_greeks(S0, 100.0, T, R, SIGMA, Q, "call", 1 << 12, seed=8)
    b = mc_greeks(S0, 100.0, T, R, SIGMA, Q, "call", 1 << 12, seed=8)
    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [{"sigma": 0.0}, {"kind": "straddle"}, {"delta_bump": 0.0}, {"vega_bump": 0.5}, {"n_paths": 1}],
)
def test_rejects_bad_inputs(kwargs):
    args = {"s0": S0, "k": 100.0, "t": T, "r": R, "sigma": SIGMA, "n_paths": 1000} | kwargs
    with pytest.raises(ValueError):
        mc_greeks(**args)
