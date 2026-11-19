"""Project 1 step 5: models beyond GBM.

1. Merton jump-diffusion: exact simulation vs the series closed form, and the smile it implies.
2. Heston semi-analytic pricer vs published benchmark values (arXiv:2606.13992, Table 4).
3. Heston smile: QE simulation at fine steps vs the semi-analytic price.
4. Discretisation bias (figure): full-truncation Euler vs Andersen QE as the step size shrinks,
   with the Feller condition badly violated so the variance keeps hitting zero.

Gated (4 sigma): items 1–3, which are guaranteed by theory. Item 4 is a measurement, not a
pass/fail property, and is skipped with --no-figure to keep CI fast.

    uv run python -m mc_engine.experiments.models
    uv run python -m mc_engine.experiments.models --no-figure
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from mc_engine.heston import HestonParams, heston_price, price_heston_mc
from mc_engine.implied_vol import implied_vol
from mc_engine.merton import MertonParams, merton_price, price_merton_mc

SEED = 20261119
Z_TOL = 4.0
STRIKES = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "heston_scheme_bias.png"

MERTON = MertonParams(sigma=0.15, lam=0.5, mu_j=-0.10, delta=0.15)
HESTON = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)
HARSH = HestonParams(kappa=0.5, theta=0.04, xi=1.0, rho=-0.9, v0=0.04)


def merton_section() -> bool:
    s0, t, r = 100.0, 1.0, 0.05
    ok = True
    print(
        f"## Merton jump-diffusion: sigma = {MERTON.sigma}, lambda = {MERTON.lam}/yr, "
        f"mean log jump = {MERTON.mu_j}, jump sd = {MERTON.delta}; S0 = 100, T = 1y\n"
    )
    print("| Strike | Series price | Exact MC (262,144 paths) | z | Implied vol |")
    print("|---:|---:|---:|---:|---:|")
    for i, k in enumerate(STRIKES):
        ref = merton_price(s0, k, t, r, MERTON)
        mc = price_merton_mc(s0, k, t, r, MERTON, n_paths=1 << 18, seed=[SEED, 1, i])
        ok &= abs(mc.z_score(ref)) < Z_TOL
        iv = implied_vol(ref, s0, k, t, r)
        print(f"| {k:.0f} | {ref:.4f} | {mc.price:.4f} | {mc.z_score(ref):+.2f} | {iv:.2%} |")
    return ok


def benchmark_section() -> bool:
    base = {"kappa": 2.0, "theta": 0.04, "xi": 0.35, "rho": -0.7, "v0": 0.04}
    cases = [
        ("v0", 0.01, 7.85571767),
        ("v0", 0.09, 10.99728901),
        ("theta", 0.01, 7.26603514),
        ("theta", 0.09, 11.56109741),
    ]
    ok = True
    print("\n## Heston semi-analytic price vs published benchmarks (arXiv:2606.13992, Table 4)\n")
    print("| Case | Benchmark | This repo | Difference |")
    print("|---|---:|---:|---:|")
    for name, val, bench in cases:
        mine = heston_price(100.0, 100.0, 1.0, 0.03, HestonParams(**(base | {name: val})))
        ok &= abs(mine - bench) < 2e-7
        print(f"| {name} = {val} | {bench:.8f} | {mine:.8f} | {mine - bench:+.1e} |")
    return ok


def heston_smile_section() -> bool:
    s0, t, r = 100.0, 1.0, 0.05
    ok = True
    print(
        f"\n## Heston smile: kappa = {HESTON.kappa}, theta = {HESTON.theta}, xi = {HESTON.xi}, "
        f"rho = {HESTON.rho}, v0 = {HESTON.v0}; S0 = 100, T = 1y\n"
    )
    print("| Strike | Semi-analytic | QE MC (128 steps, 131,072 paths) | z | Implied vol |")
    print("|---:|---:|---:|---:|---:|")
    for i, k in enumerate(STRIKES):
        ref = heston_price(s0, k, t, r, HESTON)
        mc = price_heston_mc(s0, k, t, r, HESTON, n_steps=128, n_paths=1 << 17, seed=[SEED, 2, i])
        ok &= abs(mc.z_score(ref)) < Z_TOL
        iv = implied_vol(ref, s0, k, t, r)
        print(f"| {k:.0f} | {ref:.4f} | {mc.price:.4f} | {mc.z_score(ref):+.2f} | {iv:.2%} |")
    return ok


def bias_study(t=10.0, steps_per_year=(1, 2, 4, 8, 16, 32), n=1 << 17):
    ref = heston_price(100.0, 100.0, t, 0.0, HARSH)
    rows = {"euler": [], "qe": []}
    for i, spy in enumerate(steps_per_year):
        for scheme in rows:
            rows[scheme].append(
                price_heston_mc(
                    100.0,
                    100.0,
                    t,
                    0.0,
                    HARSH,
                    n_steps=int(spy * t),
                    n_paths=n,
                    seed=[SEED, 3, i],
                    scheme=scheme,
                )
            )
    print(
        f"\n## Discretisation bias, Feller badly violated (2·kappa·theta = "
        f"{2 * HARSH.kappa * HARSH.theta:.2f} vs xi² = {HARSH.xi**2:.2f}): kappa = {HARSH.kappa}, "
        f"theta = {HARSH.theta}, xi = {HARSH.xi}, rho = {HARSH.rho}; ATM call, T = {t:.0f}y, "
        f"r = 0, {n:,} paths\n"
    )
    print(f"Semi-analytic price {ref:.4f}.\n")
    print("| Steps per year | Euler (full truncation) | bias (z) | QE | bias (z) |")
    print("|---:|---:|---:|---:|---:|")
    for spy, e, q in zip(steps_per_year, rows["euler"], rows["qe"], strict=True):
        print(
            f"| {spy} | {e.price:.4f} | {e.price - ref:+.4f} ({e.z_score(ref):+.1f}) "
            f"| {q.price:.4f} | {q.price - ref:+.4f} ({q.z_score(ref):+.1f}) |"
        )
    return steps_per_year, rows, ref


def plot(steps_per_year, rows, ref) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    x = np.array(steps_per_year, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(0, color=ink2, lw=1.0)
    for scheme, color, marker, label in (
        ("euler", blue, "o", "Full-truncation Euler"),
        ("qe", orange, "s", "Andersen QE"),
    ):
        err = np.array([r.price - ref for r in rows[scheme]])
        se = np.array([r.std_error for r in rows[scheme]])
        ax.errorbar(
            x,
            err,
            yerr=2 * se,
            color=color,
            lw=2,
            marker=marker,
            ms=5.5,
            mec=surface,
            mew=1.5,
            capsize=3,
            elinewidth=1.2,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(s) for s in steps_per_year])
    ax.set_xlabel("Time steps per year", color=ink2)
    ax.set_ylabel("Price error vs semi-analytic ($)", color=ink2)
    ax.set_title(
        "When variance keeps hitting zero, Euler's bias lingers; QE's is gone by 8 steps/yr",
        loc="left",
        color=ink,
        fontsize=11,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.035,
        f"Heston ATM call, T = 10y, Feller violated (2κθ = 0.04 < ξ² = 1) · "
        f"true price {ref:.2f} · bars ±2 SE",
        transform=ax.transAxes,
        color=ink2,
        fontsize=8.5,
    )
    ax.grid(True, which="major", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="upper right")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true", help="skip the (slow) bias study")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="The occurrence of roundoff error")
    ok_m = merton_section()
    ok_b = benchmark_section()
    ok_h = heston_smile_section()
    if not args.no_figure:
        plot(*bias_study())
        print(f"\nFigure written to {FIGURE}")
    print(
        f"\nStep 5 gate: merton {'PASS' if ok_m else 'FAIL'} · heston benchmarks "
        f"{'PASS' if ok_b else 'FAIL'} · heston smile {'PASS' if ok_h else 'FAIL'}"
    )
    if not (ok_m and ok_b and ok_h):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
