"""Brownian constructions are linear in z, so their covariance can be checked exactly, not
statistically: feeding the identity matrix returns the matrix A with W = A z, and A A^T must
equal min(t_i, t_j)."""

import numpy as np
import pytest
from scipy.stats import qmc

from mc_engine.brownian import brownian_motion, sobol_normals


@pytest.mark.parametrize("construction", ["standard", "bridge"])
@pytest.mark.parametrize("n_steps", [1, 2, 3, 5, 8, 13, 64])
def test_covariance_is_exactly_min_s_t(construction, n_steps):
    t = 2.0
    a = brownian_motion(np.eye(n_steps), t, construction)[:, 1:].T  # row i: W(t_i) loadings
    grid = np.linspace(t / n_steps, t, n_steps)
    np.testing.assert_allclose(a @ a.T, np.minimum.outer(grid, grid), atol=1e-12)


@pytest.mark.parametrize("construction", ["standard", "bridge"])
def test_paths_start_at_zero(construction):
    w = brownian_motion(np.random.default_rng(0).standard_normal((10, 16)), 1.0, construction)
    assert w.shape == (10, 17)
    assert np.all(w[:, 0] == 0.0)


def test_bridge_puts_terminal_value_in_first_coordinate():
    z = np.random.default_rng(1).standard_normal((100, 64))
    w = brownian_motion(z, 1.5, "bridge")
    np.testing.assert_allclose(w[:, -1], np.sqrt(1.5) * z[:, 0], rtol=1e-14)
    # Changing every later coordinate leaves W_T untouched.
    z2 = z.copy()
    z2[:, 1:] = 0.0
    np.testing.assert_array_equal(brownian_motion(z2, 1.5, "bridge")[:, -1], w[:, -1])


def test_rejects_unknown_construction():
    with pytest.raises(ValueError):
        brownian_motion(np.zeros((2, 4)), 1.0, "spiral")


def test_sobol_normals_are_standard_normal_marginals():
    sampler = qmc.Sobol(d=3, scramble=True, rng=np.random.default_rng(2))
    z = sobol_normals(sampler, 1 << 14)
    assert z.shape == (1 << 14, 3)
    assert np.all(np.isfinite(z))
    # QMC marginals are near-exact: errors far below the 1/sqrt(n) ~ 0.008 of random sampling.
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-3)
