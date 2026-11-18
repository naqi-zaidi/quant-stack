"""Monte Carlo checks. Every tolerance is stated in standard errors, never as a fixed decimal."""

import numpy as np
import pytest

from mc_engine import bs_price, gbm, price_european

Z_TOL = 4.0  # |z| > 4 has probability ~6e-5 under N(0, 1); seeds are fixed so this never flakes

GRID = [
    (kind, k, t, sigma)
    for kind in ("call", "put")
    for k in (80.0, 100.0, 120.0)
    for t in (0.25, 1.0)
    for sigma in (0.15, 0.40)
]


def test_matches_black_scholes_across_grid():
    seeds = np.random.SeedSequence(20261118).spawn(len(GRID))
    for (kind, k, t, sigma), seed in zip(GRID, seeds, strict=True):
        res = price_european(
            100.0, k, t, 0.05, sigma, q=0.01, kind=kind, n_paths=1 << 18, seed=seed
        )
        ref = bs_price(100.0, k, t, 0.05, sigma, q=0.01, kind=kind)
        assert abs(res.z_score(ref)) < Z_TOL, (kind, k, t, sigma, res, ref)


def test_standard_error_shrinks_as_one_over_root_n():
    small = price_european(100, 100, 1.0, 0.05, 0.2, n_paths=1 << 16, seed=1)
    large = price_european(100, 100, 1.0, 0.05, 0.2, n_paths=1 << 18, seed=2)
    assert large.std_error / small.std_error == pytest.approx(0.5, rel=0.05)


def test_discounted_terminal_price_is_a_martingale():
    s0, r, q, sigma, t, n = 100.0, 0.05, 0.02, 0.3, 2.0, 1 << 18
    rng = np.random.default_rng(7)
    x = np.exp(-(r - q) * t) * gbm.terminal(s0, r, sigma, t, rng.standard_normal(n), q=q)
    se = x.std(ddof=1) / np.sqrt(n)
    assert abs(x.mean() - s0) < Z_TOL * se


def test_chunking_does_not_change_the_estimate():
    one = price_european(100, 105, 0.5, 0.03, 0.25, n_paths=200_000, seed=11)
    many = price_european(100, 105, 0.5, 0.03, 0.25, n_paths=200_000, seed=11, chunk=7_919)
    assert many.price == pytest.approx(one.price, rel=1e-12)
    assert many.std_error == pytest.approx(one.std_error, rel=1e-9)


def test_same_seed_reproduces_exactly():
    a = price_european(100, 100, 1.0, 0.05, 0.2, n_paths=50_000, seed=3)
    b = price_european(100, 100, 1.0, 0.05, 0.2, n_paths=50_000, seed=3)
    assert a == b


def test_confidence_interval_brackets_price():
    res = price_european(100, 100, 1.0, 0.05, 0.2, n_paths=50_000, seed=3)
    lo, hi = res.ci95
    assert lo < res.price < hi
    assert hi - lo == pytest.approx(2 * 1.96 * res.std_error, rel=1e-3)


def test_paths_have_exact_log_moments():
    s0, r, q, sigma, t, n_steps, n = 100.0, 0.05, 0.01, 0.25, 1.0, 12, 1 << 17
    p = gbm.paths(s0, r, sigma, t, n_steps, n, np.random.default_rng(5), q=q)
    assert p.shape == (n, n_steps + 1)
    assert np.all(p[:, 0] == s0)
    log_ret = np.log(p[:, -1] / s0)
    mean_se = log_ret.std(ddof=1) / np.sqrt(n)
    assert abs(log_ret.mean() - (r - q - 0.5 * sigma**2) * t) < Z_TOL * mean_se
    # Sample variance of n normals has relative SE sqrt(2 / (n - 1)).
    assert log_ret.var(ddof=1) == pytest.approx(sigma**2 * t, rel=Z_TOL * np.sqrt(2 / (n - 1)))


@pytest.mark.parametrize(
    "kwargs",
    [{"s0": -1.0}, {"sigma": -0.1}, {"t": 0.0}, {"n_paths": 1}, {"kind": "digital"}],
)
def test_rejects_bad_inputs(kwargs):
    args = {"s0": 100.0, "k": 100.0, "t": 1.0, "r": 0.05, "sigma": 0.2, "n_paths": 1000} | kwargs
    with pytest.raises(ValueError):
        price_european(**args)
