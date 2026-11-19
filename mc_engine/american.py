"""American and Bermudan options: Longstaff–Schwartz Monte Carlo and a binomial reference.

Reference: a binomial tree with two standard accuracy fixes, Black–Scholes smoothing at the
last step plus Richardson extrapolation (Broadie and Detemple, 1996). Plain CRR trees
oscillate with the step count; BBSR converges smoothly enough that 2 V(2n) - V(n) is accurate
to about 1e-4 at n = 2500. The same tree prices European, Bermudan (exercise on a fixed set of
dates) and American options, so each Monte Carlo number has a like-for-like reference.

Longstaff–Schwartz (2001): simulate paths, then step backward through the exercise dates. At
each date, regress realized discounted future cash flows on polynomials in S/K, using in-the-
money paths only, and exercise where the payoff beats the fitted continuation value.

The resulting price has two biases pulling opposite ways:

* low: the fitted rule is only an approximation of the optimal one, so it under-collects;
* high: the rule is evaluated on the paths it was fitted to, so it partly knows their futures.

Pricing out of sample removes the second. The rule is fitted on one set of paths and applied
to a fresh, independent set. The result is a true lower bound, and its standard error is the
honest i.i.d. one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mc_engine import gbm
from mc_engine.black_scholes import OptionKind, bs_price
from mc_engine.paths import seed_sequence
from mc_engine.pricer import MCResult, Seed, payoff

Exercise = Literal["american", "european"] | int  # int = number of Bermudan exercise dates


# ----------------------------------------------------------------------------- binomial tree


def _bbs_tree(s0, k, t, r, sigma, q, kind, n, exercise) -> float:
    dt = t / n
    up = sigma * np.sqrt(dt)
    u, d = np.exp(up), np.exp(-up)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    if isinstance(exercise, int):
        if n % exercise:
            raise ValueError("tree steps must be a multiple of the number of Bermudan dates")
        every = n // exercise

    def can_exercise(i: int) -> bool:
        if exercise == "american":
            return True
        if exercise == "european":
            return False
        return i > 0 and i % every == 0

    i = n - 1
    s = s0 * np.exp(up * (2 * np.arange(i + 1) - i))
    v = bs_price(s, k, dt, r, sigma, q, kind)  # smoothing: last step priced in closed form
    if can_exercise(i):
        v = np.maximum(v, payoff(s, k, kind))
    for i in range(n - 2, -1, -1):
        v = disc * (p * v[1:] + (1 - p) * v[:-1])
        if can_exercise(i):
            s = s0 * np.exp(up * (2 * np.arange(i + 1) - i))
            v = np.maximum(v, payoff(s, k, kind))
    return float(v[0])


def tree_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "put",
    exercise: Exercise = "american",
    n_steps: int = 2500,
) -> float:
    """Binomial–Black–Scholes price with Richardson extrapolation: 2 V(2n) - V(n)."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if isinstance(exercise, str) and exercise not in ("american", "european"):
        raise ValueError(f"unknown exercise style {exercise!r}")
    coarse = _bbs_tree(s0, k, t, r, sigma, q, kind, n_steps, exercise)
    fine = _bbs_tree(s0, k, t, r, sigma, q, kind, 2 * n_steps, exercise)
    return 2.0 * fine - coarse


# ------------------------------------------------------------------------- Longstaff–Schwartz


@dataclass(frozen=True)
class ExercisePolicy:
    """Regression coefficients per exercise date (None where too few paths were in the money)."""

    coefs: tuple[np.ndarray | None, ...]
    degree: int


def _basis(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, degree + 1, increasing=True)  # 1, x, x^2, ..., on x = S / K


def fit_policy(
    paths: np.ndarray, k: float, r: float, dt: float, kind: OptionKind, degree: int = 3
) -> tuple[ExercisePolicy, np.ndarray]:
    """Backward induction on ``paths`` (n_paths, n_dates + 1).

    Returns the policy and the per-path cash flows discounted to time 0 (the in-sample value).
    """
    n_dates = paths.shape[1] - 1
    cash = payoff(paths[:, -1], k, kind)
    when = np.full(paths.shape[0], n_dates)  # exercise date index per path
    coefs: list[np.ndarray | None] = [None] * (n_dates - 1)
    for j in range(n_dates - 1, 0, -1):
        s = paths[:, j]
        exercise_value = payoff(s, k, kind)
        itm = exercise_value > 0
        if itm.sum() <= degree + 1:
            continue
        continuation = cash[itm] * np.exp(-r * dt * (when[itm] - j))
        basis = _basis(s[itm] / k, degree)
        beta, *_ = np.linalg.lstsq(basis, continuation, rcond=None)
        coefs[j - 1] = beta
        ex = np.zeros_like(itm)
        ex[itm] = exercise_value[itm] > basis @ beta
        cash[ex] = exercise_value[ex]
        when[ex] = j
    return ExercisePolicy(tuple(coefs), degree), cash * np.exp(-r * dt * when)


def apply_policy(
    paths: np.ndarray, policy: ExercisePolicy, k: float, r: float, dt: float, kind: OptionKind
) -> np.ndarray:
    """Forward pass with a fixed policy. Returns per-path cash flows discounted to time 0."""
    n_paths, n_cols = paths.shape
    n_dates = n_cols - 1
    value = np.zeros(n_paths)
    done = np.zeros(n_paths, dtype=bool)
    for j in range(1, n_dates):
        beta = policy.coefs[j - 1]
        if beta is None:
            continue
        s = paths[:, j]
        exercise_value = payoff(s, k, kind)
        ex = ~done & (exercise_value > 0)
        ex[ex] = exercise_value[ex] > _basis(s[ex] / k, policy.degree) @ beta
        value[ex] = exercise_value[ex] * np.exp(-r * dt * j)
        done |= ex
    last = ~done
    value[last] = payoff(paths[last, -1], k, kind) * np.exp(-r * dt * n_dates)
    return value


def lsm_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: OptionKind = "put",
    n_dates: int = 50,
    n_paths: int = 1 << 17,
    seed: Seed = 0,
    *,
    degree: int = 3,
    out_of_sample: bool = True,
) -> MCResult:
    """Longstaff–Schwartz price of a Bermudan option exercisable on ``n_dates`` equally spaced
    dates in (0, T].

    ``out_of_sample=True`` fits the rule on ``n_paths`` training paths and prices with
    ``n_paths`` fresh ones, giving a lower bound with an i.i.d. standard error.
    ``out_of_sample=False`` is the textbook in-sample estimate.
    """
    if n_dates < 1 or n_paths < 2 or degree < 1:
        raise ValueError("need n_dates >= 1, n_paths >= 2 and degree >= 1")
    train_seed, price_seed = seed_sequence(seed).spawn(2)
    dt = t / n_dates
    train = gbm.paths(s0, r, sigma, t, n_dates, n_paths, np.random.default_rng(train_seed), q=q)
    policy, in_sample = fit_policy(train, k, r, dt, kind, degree)
    if out_of_sample:
        fresh = gbm.paths(s0, r, sigma, t, n_dates, n_paths, np.random.default_rng(price_seed), q=q)
        cash = apply_policy(fresh, policy, k, r, dt, kind)
        name = "lsm_out_of_sample"
    else:
        cash, name = in_sample, "lsm_in_sample"
    return MCResult(
        float(cash.mean()), float(cash.std(ddof=1) / np.sqrt(n_paths)), n_paths, method=name
    )
