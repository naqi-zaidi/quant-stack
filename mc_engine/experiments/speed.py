"""Project 1 step 7: NumPy vs Numba throughput on three Monte Carlo kernels.

Kernels: a European call (one draw per path), an arithmetic Asian with 252 monitoring steps,
and Heston with Andersen QE over 252 steps (two draws and a branch per step).

For each: paths per second for NumPy (the package's existing vectorised code), Numba serial,
and Numba parallel, best of 3 runs after a warm-up (compile time is reported separately).
The profile column is the share of each implementation's time spent drawing normals, measured
by timing the generator alone for the same number of draws.

Speed depends on the machine, so it is reported, not gated. The gate is correctness, the
condition for the speed numbers to mean anything: Numba prices within 4 standard errors of the
reference, serial and parallel bit-identical. (Exact agreement with NumPy on identical normals
is covered in tests/test_fast.py.)

    uv run python -m mc_engine.experiments.speed
"""

from __future__ import annotations

import math
import platform
import time

import numba
import numpy as np
from numba import njit

from mc_engine import bs_price, price_european
from mc_engine.asian import price_asian
from mc_engine.fast import (
    _normal,
    _seed_state,
    price_asian_numba,
    price_european_numba,
    price_heston_numba,
)
from mc_engine.heston import HestonParams, heston_price, price_heston_mc

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
HESTON = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)
STEPS = 252
Z_TOL = 4.0


@njit
def _numba_normals(n: int, seed: np.uint64) -> float:
    r0, r1, r2, r3 = _seed_state(seed, 0)
    has, spare, acc = False, 0.0, 0.0
    for _ in range(n):
        z, r0, r1, r2, r3, has, spare = _normal(r0, r1, r2, r3, has, spare)
        acc += z
    return acc


def best_time(fn, reps: int = 3):
    fn()  # warm-up
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def rng_time(n_draws: int) -> tuple[float, float]:
    """Seconds to draw n_draws normals: NumPy (PCG64 + ziggurat), Numba (xoshiro + Box–Muller)."""
    rng = np.random.default_rng(0)
    chunk = 1 << 20

    def numpy_draws():
        left = n_draws
        while left > 0:
            rng.standard_normal(min(chunk, left))
            left -= chunk

    t_np, _ = best_time(numpy_draws)
    t_nb, _ = best_time(lambda: _numba_normals(n_draws, np.uint64(1)))
    return t_np, t_nb


def main() -> None:
    print(
        f"## Machine: {platform.processor() or platform.machine()}, "
        f"{numba.config.NUMBA_NUM_THREADS} threads available to Numba; "
        f"numpy {np.__version__}, numba {numba.__version__}\n"
    )

    t0 = time.perf_counter()
    price_european_numba(S0, K, T, R, SIGMA, n_paths=10_000, parallel=False)
    price_european_numba(S0, K, T, R, SIGMA, n_paths=10_000)
    price_asian_numba(S0, K, T, R, SIGMA, n_steps=4, n_paths=10_000, parallel=False)
    price_asian_numba(S0, K, T, R, SIGMA, n_steps=4, n_paths=10_000)
    price_heston_numba(S0, K, T, R, HESTON, n_steps=4, n_paths=10_000, parallel=False)
    price_heston_numba(S0, K, T, R, HESTON, n_steps=4, n_paths=10_000)
    print(f"One-off JIT compile of all six Numba kernels: {time.perf_counter() - t0:.1f} s\n")

    n_eu, n_path = 1 << 22, 1 << 16
    cases = [
        (
            "European call",
            n_eu,
            1,
            1,
            lambda: price_european(S0, K, T, R, SIGMA, n_paths=n_eu),
            lambda p: price_european_numba(S0, K, T, R, SIGMA, n_paths=n_eu, parallel=p),
            float(bs_price(S0, K, T, R, SIGMA)),
        ),
        (
            f"Arithmetic Asian, {STEPS} steps",
            n_path,
            STEPS,
            STEPS,
            lambda: price_asian(
                S0, K, T, R, SIGMA, n_steps=STEPS, n_paths=n_path, construction="standard"
            ),
            lambda p: price_asian_numba(
                S0, K, T, R, SIGMA, n_steps=STEPS, n_paths=n_path, parallel=p
            ),
            None,
        ),
        (
            f"Heston QE, {STEPS} steps",
            n_path,
            STEPS,
            2 * STEPS,
            lambda: price_heston_mc(S0, K, T, R, HESTON, n_steps=STEPS, n_paths=n_path),
            lambda p: price_heston_numba(
                S0, K, T, R, HESTON, n_steps=STEPS, n_paths=n_path, parallel=p
            ),
            heston_price(S0, K, T, R, HESTON),
        ),
    ]

    ok = True
    rows, profile = [], []
    for name, n, _steps, draws_per_path, numpy_fn, numba_fn, ref in cases:
        t_np, r_np = best_time(numpy_fn)
        t_se, r_se = best_time(lambda f=numba_fn: f(False))
        t_pa, r_pa = best_time(lambda f=numba_fn: f(True))
        ok &= (r_se.price, r_se.std_error) == (r_pa.price, r_pa.std_error)
        if ref is None:  # Asian: compare the two independent implementations with each other
            z = (r_pa.price - r_np.price) / math.hypot(r_pa.std_error, r_np.std_error)
        else:
            z = r_pa.z_score(ref)
        ok &= abs(z) < Z_TOL
        rows.append((name, n / t_np, n / t_se, n / t_pa, z))
        g_np, g_nb = rng_time(n * draws_per_path)
        profile.append((name, g_np / t_np, g_nb / t_se))

    print(
        "| Kernel | NumPy | Numba serial | Numba parallel | Parallel vs NumPy | z (correctness) |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    for name, a, b, c, z in rows:
        unit, scale = ("M", 1e6) if a > 1e6 else ("k", 1e3)
        print(
            f"| {name} | {a / scale:,.0f}{unit} paths/s | {b / scale:,.0f}{unit} | "
            f"{c / scale:,.0f}{unit} | {c / a:.2f}× | {z:+.2f} |"
        )

    print("\nShare of time spent drawing normals (the rest is the model itself):\n")
    print("| Kernel | NumPy (PCG64 + ziggurat) | Numba (xoshiro256** + Box–Muller) |")
    print("|---|---:|---:|")
    for name, a, b in profile:
        print(f"| {name} | {a:.0%} | {b:.0%} |")

    print(f"\nSpeed gate (correctness of the benchmarked kernels): {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
