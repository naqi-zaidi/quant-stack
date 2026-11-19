"""Numba kernels: exact agreement with NumPy on identical random numbers, unbiased prices,
bit-identical serial vs parallel runs, and a sane generator."""

import numpy as np
import pytest
from numba import njit, set_num_threads

from mc_engine import bs_price
from mc_engine.asian import price_asian
from mc_engine.fast import (
    KERNELS,
    _normal,
    _seed_state,
    price_asian_numba,
    price_european_numba,
    price_heston_numba,
)
from mc_engine.heston import HestonParams, heston_price, simulate_log_terminal
from mc_engine.paths import gbm_paths_from_normals

S0, K, T, R, Q, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.01, 0.2
P = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
Z_TOL = 4.0


@pytest.mark.parametrize("parallel", ["serial", "parallel"])
def test_asian_kernel_matches_numpy_on_the_same_normals(parallel):
    z = np.random.default_rng(1).standard_normal((2048, 52))
    numba_pay = KERNELS["asian_from_normals"][parallel](S0, K, T, R, SIGMA, Q, True, z)
    paths = gbm_paths_from_normals(S0, R, SIGMA, T, Q, z, "standard")
    numpy_pay = np.exp(-R * T) * np.maximum(paths[:, 1:].mean(axis=1) - K, 0.0)
    np.testing.assert_allclose(numba_pay, numpy_pay, rtol=1e-12, atol=1e-12)


class _ReplayRNG:
    """Feeds pre-drawn normals to the NumPy Heston code in the order it requests them."""

    def __init__(self, zv, zx):
        self._cols = [c for j in range(zv.shape[1]) for c in (zv[:, j], zx[:, j])]

    def standard_normal(self, n):
        return self._cols.pop(0).copy()


@pytest.mark.parametrize("parallel", ["serial", "parallel"])
def test_heston_kernel_matches_numpy_on_the_same_normals(parallel):
    rng = np.random.default_rng(2)
    zv, zx = rng.standard_normal((2, 4096, 24))
    numba_pay = KERNELS["heston_from_normals"][parallel](
        S0, K, T, R, Q, P.kappa, P.theta, P.xi, P.rho, P.v0, True, zv, zx
    )
    x = simulate_log_terminal(S0, T, R, P, Q, 24, 4096, _ReplayRNG(zv, zx), "qe")
    numpy_pay = np.exp(-R * T) * np.maximum(np.exp(x) - K, 0.0)
    np.testing.assert_allclose(numba_pay, numpy_pay, rtol=1e-10, atol=1e-10)


def test_european_is_unbiased():
    for kind in ("call", "put"):
        res = price_european_numba(S0, 95.0, T, R, SIGMA, Q, kind, 1 << 20, seed=3)
        ref = float(bs_price(S0, 95.0, T, R, SIGMA, Q, kind))
        assert abs(res.z_score(ref)) < Z_TOL, (kind, res, ref)


def test_heston_is_unbiased():
    res = price_heston_numba(S0, K, T, R, P, Q, n_steps=64, n_paths=1 << 17, seed=4)
    assert abs(res.z_score(heston_price(S0, K, T, R, P, Q))) < Z_TOL


def test_asian_matches_a_high_precision_reference():
    ref = price_asian(
        S0, K, T, R, SIGMA, Q, "call", 12, 1 << 16, seed=5, control_variate=True, sampler="sobol"
    )
    res = price_asian_numba(S0, K, T, R, SIGMA, Q, n_steps=12, n_paths=1 << 18, seed=6)
    assert abs((res.price - ref.price) / np.hypot(res.std_error, ref.std_error)) < Z_TOL


def test_serial_and_parallel_are_bit_identical():
    for fn, kwargs in (
        (price_european_numba, {"n_paths": 50_000}),
        (price_asian_numba, {"n_steps": 16, "n_paths": 20_000}),
        (price_heston_numba, {"n_steps": 16, "n_paths": 20_000}),
    ):
        args = (S0, K, T, R, P) if fn is price_heston_numba else (S0, K, T, R, SIGMA)
        a = fn(*args, seed=7, parallel=False, **kwargs)
        b = fn(*args, seed=7, parallel=True, **kwargs)
        assert (a.price, a.std_error) == (b.price, b.std_error), fn.__name__


def test_results_do_not_depend_on_thread_count():
    set_num_threads(1)
    a = price_heston_numba(S0, K, T, R, P, n_steps=16, n_paths=30_000, seed=8)
    set_num_threads(2)
    b = price_heston_numba(S0, K, T, R, P, n_steps=16, n_paths=30_000, seed=8)
    assert (a.price, a.std_error) == (b.price, b.std_error)


@njit
def _draw(n, seed, block):
    out = np.empty(n)
    r0, r1, r2, r3 = _seed_state(seed, block)
    has, spare = False, 0.0
    for i in range(n):
        out[i], r0, r1, r2, r3, has, spare = _normal(r0, r1, r2, r3, has, spare)
    return out


def test_generator_produces_standard_normals():
    z = _draw(1 << 21, 12345, 0)
    n = z.size
    assert abs(z.mean()) < 4 / np.sqrt(n)
    assert abs(z.var() - 1) < 4 * np.sqrt(2 / n)
    assert abs(np.mean(z**4) - 3) < 4 * np.sqrt(96 / n)  # Var(Z^4) = 105 - 9 = 96


def test_blocks_get_independent_streams():
    a, b = _draw(1 << 16, 12345, 0), _draw(1 << 16, 12345, 1)
    assert abs(np.corrcoef(a, b)[0, 1]) < 4 / np.sqrt(a.size)
