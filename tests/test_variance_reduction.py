"""Variance-reduced estimators: unbiased against Black–Scholes, honest error bars, real gains."""

import numpy as np
import pytest

from mc_engine import bs_price, price_european

S0, R, Q = 100.0, 0.05, 0.01
Z_TOL = 4.0
N = 1 << 16

METHODS = [
    ("antithetic", {}),
    ("control_variate", {}),
    ("sobol", {}),
    ("sobol", {"n_steps": 32, "construction": "standard"}),
    ("sobol", {"n_steps": 32, "construction": "bridge"}),
]
GRID = [(kind, k) for kind in ("call", "put") for k in (80.0, 100.0, 120.0)]


@pytest.mark.parametrize(("method", "opts"), METHODS)
def test_unbiased_against_black_scholes(method, opts):
    seeds = np.random.SeedSequence([7, len(opts)]).spawn(len(GRID))
    for (kind, k), seed in zip(GRID, seeds, strict=True):
        res = price_european(S0, k, 1.0, R, 0.25, Q, kind, N, seed, method=method, **opts)
        ref = float(bs_price(S0, k, 1.0, R, 0.25, Q, kind))
        assert abs(res.z_score(ref)) < Z_TOL, (method, opts, kind, k, res, ref)


@pytest.mark.parametrize(("method", "opts"), METHODS)
def test_same_seed_reproduces(method, opts):
    a = price_european(S0, 100, 1.0, R, 0.2, n_paths=1 << 12, seed=5, method=method, **opts)
    b = price_european(S0, 100, 1.0, R, 0.2, n_paths=1 << 12, seed=5, method=method, **opts)
    assert a == b


def _se(method, k=90.0, **opts):
    return price_european(S0, k, 1.0, R, 0.2, Q, "call", N, 11, method=method, **opts).std_error


def test_every_method_beats_plain_on_an_in_the_money_call():
    # Conservative floors on the variance-reduction factor (SE_plain / SE_method)^2.
    plain = _se("plain")
    assert (plain / _se("antithetic")) ** 2 > 2
    assert (plain / _se("control_variate")) ** 2 > 5
    assert (plain / _se("sobol")) ** 2 > 100


def test_bridge_keeps_the_qmc_gain_that_the_standard_construction_loses():
    opts = {"n_steps": 64}
    standard = _se("sobol", construction="standard", **opts)
    bridge = _se("sobol", construction="bridge", **opts)
    assert bridge < standard / 5


def test_one_step_sobol_is_the_same_for_both_constructions():
    a = price_european(
        S0, 100, 1.0, R, 0.2, n_paths=N, seed=3, method="sobol", construction="standard"
    )
    b = price_european(
        S0, 100, 1.0, R, 0.2, n_paths=N, seed=3, method="sobol", construction="bridge"
    )
    assert a.price == b.price


def test_rqmc_uses_student_t_quantiles():
    res = price_european(S0, 100, 1.0, R, 0.2, n_paths=N, seed=3, method="sobol", n_replicates=8)
    assert res.dof == 7
    lo, hi = res.ci95
    assert (hi - lo) / 2 == pytest.approx(2.3646 * res.std_error, rel=1e-4)  # t_{0.975, 7}


@pytest.mark.parametrize(
    ("method", "n_paths", "opts"),
    [
        ("antithetic", 1001, {}),
        ("sobol", 3000, {}),  # 3000 / 16 is not a power of 2
        ("sobol", 1 << 12, {"n_replicates": 1}),
        ("sobol", 1 << 12, {"n_steps": 0}),
        ("control_variate", 1, {}),
        ("quantum", 1 << 12, {}),
    ],
)
def test_rejects_bad_configurations(method, n_paths, opts):
    with pytest.raises(ValueError):
        price_european(S0, 100, 1.0, R, 0.2, n_paths=n_paths, method=method, **opts)


def test_plain_rejects_method_options():
    with pytest.raises(TypeError):
        price_european(S0, 100, 1.0, R, 0.2, n_paths=1000, n_steps=4)
