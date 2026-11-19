"""Project 1 step 4: path-dependent payoffs, each checked against a reference that is not
itself Monte Carlo wherever one exists.

1. Asian: geometric MC vs its closed form, then arithmetic variance-reduction factors
   from repeated independent runs (plain, geometric control variate, Sobol + bridge, both).
2. Barrier: down-and-out call vs the continuous-monitoring closed form as the grid refines,
   naive monitoring vs the Brownian-bridge crossing correction.
3. American put: Longstaff–Schwartz in and out of sample vs a binomial tree with the same 50
   exercise dates. Out of sample must be a lower bound within 1% of the tree.

All checks use 4-sigma bands; the script exits non-zero if any fails.

    uv run python -m mc_engine.experiments.path_dependent
    uv run python -m mc_engine.experiments.path_dependent --no-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mc_engine import bs_price
from mc_engine.american import lsm_price, tree_price
from mc_engine.asian import geometric_asian_price, price_asian
from mc_engine.barrier import down_and_out_call_price, price_down_and_out_call

SEED = 20261118
Z_TOL = 4.0
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "barrier_monitoring_bias.png"


def asian_section() -> bool:
    s0, k, t, r, sigma, q, n_steps = 100.0, 100.0, 1.0, 0.05, 0.25, 0.0, 12
    runs, n = 100, 1 << 14
    ok = True
    print(
        f"## Asian call, monthly averaging (n = {n_steps}), S0 = K = {s0:.0f}, "
        f"sigma = {sigma:.0%}, T = {t:.0f}y\n"
    )
    geo_ref = geometric_asian_price(s0, k, t, r, sigma, q, "call", n_steps)
    print("| Geometric average | Price | z vs closed form |")
    print("|---|---:|---:|")
    print(f"| Closed form | {geo_ref:.4f} | |")
    for i, sampler in enumerate(("pseudo", "sobol")):
        res = price_asian(
            s0,
            k,
            t,
            r,
            sigma,
            q,
            "call",
            n_steps,
            1 << 18,
            [SEED, 1, i],
            average="geometric",
            sampler=sampler,
        )
        ok &= abs(res.z_score(geo_ref)) < Z_TOL
        print(f"| MC, {sampler}, 262,144 paths | {res.price:.4f} | {res.z_score(geo_ref):+.2f} |")

    best = price_asian(
        s0,
        k,
        t,
        r,
        sigma,
        q,
        "call",
        n_steps,
        1 << 20,
        [SEED, 2],
        control_variate=True,
        sampler="sobol",
    )
    configs = [
        ("Plain MC", {}),
        ("Geometric control variate", {"control_variate": True}),
        ("Sobol + Brownian bridge", {"sampler": "sobol"}),
        ("Sobol + bridge + control variate", {"sampler": "sobol", "control_variate": True}),
    ]
    variances = {}
    print(
        f"\nArithmetic average: reference {best.price:.5f} ± {best.std_error:.5f} "
        "(Sobol + bridge + control variate, 1,048,576 paths).\n"
    )
    print(f"| Method ({n:,} paths) | VRF vs plain | worst \\|z\\| vs reference over {runs} runs |")
    print("|---|---:|---:|")
    for ci, (label, opts) in enumerate(configs):
        res = [
            price_asian(s0, k, t, r, sigma, q, "call", n_steps, n, s, **opts)
            for s in np.random.SeedSequence([SEED, 3, ci]).spawn(runs)
        ]
        prices = np.array([x.price for x in res])
        z = np.array([(x.price - best.price) / np.hypot(x.std_error, best.std_error) for x in res])
        # Each run is one check; with 400 checks at 4 sigma the family-wise false alarm is ~3%,
        # so the per-run bound is 4.5 sigma here (~0.3% overall).
        ok &= np.abs(z).max() < 4.5
        variances[label] = prices.var(ddof=1)
        vrf = variances["Plain MC"] / variances[label]
        print(f"| {label} | {vrf:,.0f}× | {np.abs(z).max():.2f} |")
    return ok


def barrier_section():
    s0, k, b, t, r, sigma, q = 100.0, 100.0, 90.0, 1.0, 0.05, 0.25, 0.0
    n = 1 << 17
    ref = down_and_out_call_price(s0, k, b, t, r, sigma, q)
    vanilla = float(bs_price(s0, k, t, r, sigma, q, "call"))
    steps = [4, 8, 16, 32, 64, 128, 256]
    rows = {"naive": [], "bridge": []}
    ok = True
    for i, m in enumerate(steps):
        for name, corr in (("naive", False), ("bridge", True)):
            res = price_down_and_out_call(
                s0, k, b, t, r, sigma, q, m, n, [SEED, 4, i, corr], bridge_correction=corr
            )
            rows[name].append(res)
            if corr:
                ok &= abs(res.z_score(ref)) < Z_TOL
    print(
        f"\n## Down-and-out call, continuous monitoring: S0 = K = {s0:.0f}, B = {b:.0f}, "
        f"sigma = {sigma:.0%}, T = {t:.0f}y, {n:,} paths\n"
    )
    print(f"Closed form {ref:.4f} (vanilla call {vanilla:.4f}).\n")
    # Broadie–Glasserman–Kou: a barrier checked on m dates prices like a continuous one moved
    # down to B exp(-0.5826 sigma sqrt(T/m)). This predicts what the naive estimator converges to.
    bgk = [
        down_and_out_call_price(s0, k, b * np.exp(-0.5826 * sigma * np.sqrt(t / m)), t, r, sigma, q)
        for m in steps
    ]
    print(
        "| Grid steps | Naive MC | BGK discrete-monitoring formula | Naive − continuous (z) "
        "| Bridge-corrected MC | z |"
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for m, a, g, c in zip(steps, rows["naive"], bgk, rows["bridge"], strict=True):
        print(
            f"| {m} | {a.price:.4f} | {g:.4f} | {a.price - ref:+.4f} ({a.z_score(ref):+.0f}) "
            f"| {c.price:.4f} | {c.z_score(ref):+.2f} |"
        )
    bias = np.array([a.price - ref for a in rows["naive"]])
    slope = np.polyfit(np.log(steps), np.log(bias), 1)[0]
    bgk_slope = np.polyfit(np.log(steps), np.log(np.array(bgk) - ref), 1)[0]
    print(
        f"\nNaive bias slope in log-log: {slope:.2f} measured, {bgk_slope:.2f} predicted by BGK "
        "over the same grids (−0.5 only in the fine-grid limit)."
    )
    return ok, steps, rows, ref, slope, bgk


def american_section() -> bool:
    k, r, t, n_dates, n = 40.0, 0.06, 1.0, 50, 1 << 16
    ok = True
    print(
        f"\n## American put, K = {k:.0f}, r = {r:.0%}, T = {t:.0f}y "
        "(the Longstaff–Schwartz test grid)\n"
    )
    print(
        "LSM uses 50 exercise dates and cubic polynomials in S/K, fit on in-the-money paths. "
        f"{n:,} paths, plus {n:,} fresh paths for out of sample.\n"
    )
    print(
        "| S0 | σ | European (BS) | Bermudan-50 (tree) | American (tree) | LSM in-sample "
        "| LSM out-of-sample | OOS − tree |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, (s0, sigma) in enumerate(
        [(36, 0.2), (36, 0.4), (40, 0.2), (40, 0.4), (44, 0.2), (44, 0.4)]
    ):
        euro = float(bs_price(s0, k, t, r, sigma, kind="put"))
        berm = tree_price(s0, k, t, r, sigma, exercise=n_dates)
        amer = tree_price(s0, k, t, r, sigma)
        ins = lsm_price(
            s0, k, t, r, sigma, n_dates=n_dates, n_paths=n, seed=[SEED, 5, i], out_of_sample=False
        )
        oos = lsm_price(s0, k, t, r, sigma, n_dates=n_dates, n_paths=n, seed=[SEED, 5, i])
        ok &= oos.price - berm < Z_TOL * oos.std_error
        ok &= berm - oos.price < 0.01 * berm + Z_TOL * oos.std_error
        print(
            f"| {s0} | {sigma:.1f} | {euro:.4f} | {berm:.4f} | {amer:.4f} | {ins.price:.4f} "
            f"| {oos.price:.4f} ± {oos.std_error:.4f} | {oos.price - berm:+.4f} |"
        )
    return ok


def plot(steps, rows, ref, slope, bgk) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    x = np.array(steps, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(0, color=ink2, lw=1.0)
    ax.plot(
        x,
        np.array(bgk) - ref,
        color=ink2,
        lw=1.2,
        ls=(0, (4, 3)),
        zorder=1,
        label="BGK formula for grid-only monitoring (theory)",
    )
    for name, color, marker, label in (
        ("naive", blue, "o", f"Naive: check the barrier on the grid only (slope {slope:.2f})"),
        ("bridge", orange, "s", "Brownian-bridge crossing correction"),
    ):
        err = np.array([res.price - ref for res in rows[name]])
        se = np.array([res.std_error for res in rows[name]])
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
            label=label.replace("-0", "−0"),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(s) for s in steps])
    ax.set_xlabel("Grid steps per year", color=ink2)
    ax.set_ylabel("Price error vs closed form ($)", color=ink2)
    ax.set_title(
        "A path can cross the barrier between grid points; the bridge correction counts it",
        loc="left",
        color=ink,
        fontsize=11,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.035,
        "Down-and-out call, S0 = K = 100, B = 90, σ = 25%, T = 1y · "
        "131,072 paths per point · bars ±2 SE",
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
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    ok_asian = asian_section()
    ok_barrier, steps, rows, ref, slope, bgk = barrier_section()
    ok_american = american_section()
    if not args.no_figure:
        plot(steps, rows, ref, slope, bgk)
        print(f"\nFigure written to {FIGURE}")
    passed = ok_asian and ok_barrier and ok_american
    print(
        f"\nStep 4 gate: asian {'PASS' if ok_asian else 'FAIL'} · barrier "
        f"{'PASS' if ok_barrier else 'FAIL'} · american {'PASS' if ok_american else 'FAIL'}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
