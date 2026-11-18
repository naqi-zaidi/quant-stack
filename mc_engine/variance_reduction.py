"""Variance-reduced Monte Carlo estimators for European options under GBM.

Every estimator returns an ``MCResult`` whose ``n_paths`` counts payoff evaluations, so methods
are compared at equal cost, and whose standard error is computed the way that method's
variance actually behaves:

* antithetic: the error comes from the pair averages, not the 2N individual payoffs, because
  the two halves of a pair are deliberately correlated.
* control variate: beta comes from an independent pilot run. Fitting beta on the main sample
  would add an O(1/N) bias.
* randomized QMC (Sobol): a single Sobol run has no usable error estimate, so the method runs
  independent scramblings and reports the spread of their means, with Student's t quantiles.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from mc_engine import gbm
from mc_engine.black_scholes import OptionKind
from mc_engine.brownian import Construction, brownian_motion, sobol_normals
from mc_engine.pricer import MCResult, Seed, _RunningMoments, payoff

QMC_CHUNK = 1 << 16


def _seed_sequence(seed: Seed) -> np.random.SeedSequence:
    return seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)


def antithetic(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    kind: OptionKind,
    n_paths: int,
    seed: Seed,
    chunk: int,
) -> MCResult:
    """Pairs each draw z with -z and averages the two payoffs.

    It works when the payoff is monotone in z, because then f(z) and f(-z) are negatively
    correlated and their average varies less than either alone. It also halves the number of
    normals drawn.
    """
    if n_paths < 4 or n_paths % 2:
        raise ValueError("antithetic needs an even n_paths of at least 4")
    rng = np.random.default_rng(seed)
    discount = np.exp(-r * t)
    moments = _RunningMoments()
    pairs = n_paths // 2
    while pairs > 0:
        m = min(max(chunk // 2, 1), pairs)
        z = rng.standard_normal(m)
        up = payoff(gbm.terminal(s0, r, sigma, t, z, q=q), k, kind)
        down = payoff(gbm.terminal(s0, r, sigma, t, -z, q=q), k, kind)
        moments.update(0.5 * discount * (up + down))
        pairs -= m
    return MCResult(moments.mean, moments.std_error(), n_paths, method="antithetic")


def control_variate(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    kind: OptionKind,
    n_paths: int,
    seed: Seed,
    chunk: int,
    *,
    pilot_paths: int = 10_000,
) -> MCResult:
    """Uses the discounted terminal price X = e^{-rT} S_T, whose mean S0 e^{-qT} is known.

    Estimator: mean(Y - beta (X - E[X])), with beta = Cov(X, Y) / Var(X) from an independent
    pilot run. Variance falls by a factor of 1 / (1 - rho^2), where rho = corr(X, Y), so the
    gain is large for in-the-money calls (payoff almost linear in S_T) and small far
    out of the money.
    """
    if n_paths < 2 or pilot_paths < 2:
        raise ValueError("n_paths and pilot_paths must be at least 2")
    pilot_seed, main_seed = _seed_sequence(seed).spawn(2)
    discount = np.exp(-r * t)
    mean_x = s0 * np.exp(-q * t)

    rng = np.random.default_rng(pilot_seed)
    s_t = gbm.terminal(s0, r, sigma, t, rng.standard_normal(pilot_paths), q=q)
    x, y = discount * s_t, discount * payoff(s_t, k, kind)
    var_x = x.var(ddof=1)
    beta = float(np.cov(x, y)[0, 1] / var_x) if var_x > 0 else 0.0

    rng = np.random.default_rng(main_seed)
    moments = _RunningMoments()
    remaining = n_paths
    while remaining > 0:
        m = min(chunk, remaining)
        s_t = gbm.terminal(s0, r, sigma, t, rng.standard_normal(m), q=q)
        moments.update(discount * payoff(s_t, k, kind) - beta * (discount * s_t - mean_x))
        remaining -= m
    return MCResult(moments.mean, moments.std_error(), n_paths, method="control_variate")


def sobol_rqmc(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float,
    kind: OptionKind,
    n_paths: int,
    seed: Seed,
    chunk: int,
    *,
    n_replicates: int = 16,
    n_steps: int = 1,
    construction: Construction = "bridge",
) -> MCResult:
    """Randomized quasi-Monte Carlo with scrambled Sobol points.

    Splits n_paths into ``n_replicates`` independently scrambled Sobol runs of equal power-of-2
    size. Each run's mean is unbiased, so the reported price is their average and the standard
    error is their spread over sqrt(n_replicates), with n_replicates - 1 degrees of freedom.

    ``n_steps > 1`` builds full Brownian paths, which is what path-dependent payoffs need. With
    the ``standard`` construction S_T then depends on every Sobol dimension. With the
    ``bridge`` construction it depends on the first dimension only, which keeps the QMC gain.
    ``chunk`` is unused: Sobol draws stay in power-of-2 blocks of ``QMC_CHUNK``.
    """
    del chunk
    if n_replicates < 2:
        raise ValueError("n_replicates must be at least 2 to estimate a standard error")
    per_replicate, rem = divmod(n_paths, n_replicates)
    if rem or per_replicate < 2 or per_replicate & (per_replicate - 1):
        raise ValueError(
            "n_paths / n_replicates must be a power of 2 (Sobol balance), "
            f"got {n_paths} / {n_replicates}"
        )
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    discount = np.exp(-r * t)
    sqrt_t = np.sqrt(t)
    means = np.empty(n_replicates)
    for i, child in enumerate(_seed_sequence(seed).spawn(n_replicates)):
        sampler = qmc.Sobol(d=n_steps, scramble=True, rng=np.random.default_rng(child))
        total, left = 0.0, per_replicate
        while left > 0:
            m = min(QMC_CHUNK, left)
            z = sobol_normals(sampler, m)
            if n_steps == 1:
                w_t = sqrt_t * z[:, 0]
            else:
                w_t = brownian_motion(z, t, construction)[:, -1]
            s_t = gbm.terminal(s0, r, sigma, t, w_t / sqrt_t, q=q)
            total += float(payoff(s_t, k, kind).sum())
            left -= m
        means[i] = discount * total / per_replicate
    return MCResult(
        price=float(means.mean()),
        std_error=float(means.std(ddof=1) / np.sqrt(n_replicates)),
        n_paths=n_paths,
        method="sobol" if n_steps == 1 else f"sobol_{construction}_{n_steps}",
        dof=n_replicates - 1,
    )
