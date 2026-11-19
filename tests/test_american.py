"""American / Bermudan pricing: the tree is checked against closed forms and no-arbitrage
facts; Longstaff–Schwartz is checked against the tree for the same exercise dates."""

import numpy as np
import pytest

from mc_engine import bs_price
from mc_engine.american import lsm_price, tree_price

# The classic Longstaff–Schwartz (2001) test contract.
S0, K, T, R, SIGMA = 36.0, 40.0, 1.0, 0.06, 0.20


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize(("s0", "q"), [(36.0, 0.0), (44.0, 0.02)])
def test_tree_european_matches_black_scholes(kind, s0, q):
    tree = tree_price(s0, K, T, R, SIGMA, q, kind, "european")
    assert tree == pytest.approx(float(bs_price(s0, K, T, R, SIGMA, q, kind)), abs=2e-5)


def test_american_call_without_dividends_is_never_exercised_early():
    # Merton: with q = 0, early exercise of a call is never optimal, so American = European.
    assert tree_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call") == pytest.approx(
        float(bs_price(100, 100, 1.0, 0.05, 0.2)), abs=2e-5
    )


def test_tree_has_converged():
    coarse = tree_price(S0, K, T, R, SIGMA, n_steps=1250)
    fine = tree_price(S0, K, T, R, SIGMA, n_steps=2500)
    assert abs(fine - coarse) < 1e-4


def test_exercise_rights_are_ordered():
    european = tree_price(S0, K, T, R, SIGMA, exercise="european")
    bermudan_10 = tree_price(S0, K, T, R, SIGMA, exercise=10)
    bermudan_50 = tree_price(S0, K, T, R, SIGMA, exercise=50)
    american = tree_price(S0, K, T, R, SIGMA)
    assert european < bermudan_10 < bermudan_50 < american
    # One exercise date, at maturity, is just a European option.
    assert tree_price(S0, K, T, R, SIGMA, exercise=1) == pytest.approx(european, abs=1e-9)


def test_lsm_out_of_sample_is_a_tight_lower_bound():
    ref = tree_price(S0, K, T, R, SIGMA, exercise=50)
    res = lsm_price(S0, K, T, R, SIGMA, n_dates=50, n_paths=1 << 17, seed=1)
    assert res.price - ref < 4 * res.std_error  # never meaningfully above the true value
    assert ref - res.price < 0.01 * ref + 4 * res.std_error  # policy within 1% of optimal


def test_lsm_beats_holding_to_maturity():
    european = float(bs_price(S0, K, T, R, SIGMA, kind="put"))
    res = lsm_price(S0, K, T, R, SIGMA, n_paths=1 << 16, seed=2)
    assert (res.price - european) / res.std_error > 20


def test_lsm_same_seed_reproduces():
    a = lsm_price(S0, K, T, R, SIGMA, n_paths=1 << 12, seed=3)
    b = lsm_price(S0, K, T, R, SIGMA, n_paths=1 << 12, seed=3)
    assert a == b


def test_bermudan_dates_must_divide_tree_steps():
    with pytest.raises(ValueError):
        tree_price(S0, K, T, R, SIGMA, exercise=7, n_steps=100)


@pytest.mark.parametrize("kwargs", [{"n_dates": 0}, {"n_paths": 1}, {"degree": 0}])
def test_lsm_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError):
        lsm_price(S0, K, T, R, SIGMA, **kwargs)


def test_policy_value_never_exceeds_perfect_foresight():
    # A sanity bound: no exercise rule can beat exercising at each path's best date.
    from mc_engine import gbm
    from mc_engine.pricer import payoff

    rng = np.random.default_rng(4)
    paths = gbm.paths(S0, R, SIGMA, T, 50, 1 << 14, rng)
    dt = T / 50
    disc = np.exp(-R * dt * np.arange(51))
    foresight = (payoff(paths, K, "put") * disc).max(axis=1).mean()
    res = lsm_price(S0, K, T, R, SIGMA, n_paths=1 << 14, seed=4)
    assert res.price < foresight
