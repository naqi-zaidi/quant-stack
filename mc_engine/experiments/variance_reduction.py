"""Project 1 step 2: how much does each variance-reduction method actually buy?

Each number comes from repeated independent runs, never from a method's own reported error:

1. Variance-reduction factor (VRF) = Var(plain estimates) / Var(method estimates) at equal
   path count, measured over RUNS independent runs per contract.
2. Calibration: each method's reported standard error must match the observed spread
   (mean SE^2 / observed variance near 1, z-score coverage near nominal). A method whose error
   bars are wrong fails here, and the script exits non-zero so CI catches it. There are 12
   checks (4 methods x 3), so each band is 4 sigma wide: about a 0.1% chance that a correct
   method fails on some check. At 3 sigma that chance is about 3%, which tripped once on a
   correct estimator during development. tests/test_calibration_gate.py shows the 4-sigma gate
   still rejects error bars that are 20% too small or too large.
3. Efficiency = VRF x (plain time / method time), i.e. the variance cut at equal wall-clock.
4. Brownian bridge: the same European priced from 64-step Sobol paths, with the standard
   construction and with the bridge.
5. Figure: RMSE vs N for every method on log-log axes.

    uv run python -m mc_engine.experiments.variance_reduction            # tables + figure
    uv run python -m mc_engine.experiments.variance_reduction --no-figure
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy import stats

from mc_engine import bs_price, price_european

S0, T, R, Q, SIGMA = 100.0, 1.0, 0.05, 0.0, 0.20
N = 1 << 14
RUNS = 200
SEED = 20261118
BRIDGE_STEPS = 64
BRIDGE_RUNS = 50

CONTRACTS = [("call", 80.0), ("call", 100.0), ("call", 130.0), ("put", 100.0)]
METHODS = ["plain", "antithetic", "control_variate", "sobol"]
LABELS = {
    "plain": "Plain MC",
    "antithetic": "Antithetic",
    "control_variate": "Control variate (S_T)",
    "sobol": "Sobol RQMC (16 scramblings)",
}
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "variance_reduction.png"

Z_BAND = 4.0  # band half-width in standard deviations, per calibration check
P_INSIDE_1, P_INSIDE_2 = 0.682689, 0.954500  # P(|Z| < 1), P(|Z| < 2) for Z ~ N(0, 1)


def calibration_ok(z: np.ndarray, se2_over_var: np.ndarray, runs: int) -> bool:
    """True when z-scores and SE-to-variance ratios are consistent with honest error bars.

    ``z`` pools every run's z-score; ``se2_over_var`` holds one ratio per contract, each
    computed from ``runs`` independent runs.
    """
    n = z.size
    in1, in2 = np.mean(np.abs(z) < 1), np.mean(np.abs(z) < 2)
    ok1 = abs(in1 - P_INSIDE_1) <= Z_BAND * np.sqrt(P_INSIDE_1 * (1 - P_INSIDE_1) / n)
    ok2 = abs(in2 - P_INSIDE_2) <= Z_BAND * np.sqrt(P_INSIDE_2 * (1 - P_INSIDE_2) / n)
    # A sample variance over `runs` draws has relative sd sqrt(2 / (runs - 1)).
    ratio_sd = np.sqrt(2 / (runs - 1)) / np.sqrt(len(se2_over_var))
    ok_ratio = abs(np.mean(se2_over_var) - 1) <= Z_BAND * ratio_sd
    return bool(ok1 and ok2 and ok_ratio)


def _runs(method, kind, k, n, runs, seed, **opts):
    """Independent runs of one method on one contract: arrays of prices, SEs and the dof."""
    out = [
        price_european(S0, k, T, R, SIGMA, Q, kind, n, s, method=method, **opts)
        for s in np.random.SeedSequence(seed).spawn(runs)
    ]
    return np.array([o.price for o in out]), np.array([o.std_error for o in out]), out[0].dof


def vrf_and_calibration():
    var = {}
    calib = {m: {"z": [], "se2": [], "var": []} for m in METHODS}
    for ci, (kind, k) in enumerate(CONTRACTS):
        ref = float(bs_price(S0, k, T, R, SIGMA, Q, kind))
        for mi, method in enumerate(METHODS):
            prices, ses, dof = _runs(method, kind, k, N, RUNS, [SEED, ci, mi])
            var[(kind, k, method)] = prices.var(ddof=1)
            z = (prices - ref) / ses
            if dof is not None:  # map t-distributed scores onto the normal scale
                z = stats.norm.ppf(stats.t.cdf(z, dof))
            calib[method]["z"].append(z)
            calib[method]["se2"].append(np.mean(ses**2))
            calib[method]["var"].append(prices.var(ddof=1))
    return var, calib


def timings(n=1 << 18, repeats=5):
    out = {}
    for method in METHODS:
        price_european(S0, 100.0, T, R, SIGMA, Q, "call", n, 0, method=method)  # warm up
        best = np.inf
        for i in range(repeats):
            t0 = time.perf_counter()
            price_european(S0, 100.0, T, R, SIGMA, Q, "call", n, i, method=method)
            best = min(best, time.perf_counter() - t0)
        out[method] = best
    return out


def bridge_table(var_plain_atm):
    rows = []
    for label, opts in [
        ("Sobol, 1 step (terminal draw only)", {}),
        (
            f"Sobol, {BRIDGE_STEPS} steps, standard construction",
            {"n_steps": BRIDGE_STEPS, "construction": "standard"},
        ),
        (
            f"Sobol, {BRIDGE_STEPS} steps, Brownian bridge",
            {"n_steps": BRIDGE_STEPS, "construction": "bridge"},
        ),
    ]:
        prices, _, _ = _runs("sobol", "call", 100.0, N, BRIDGE_RUNS, [SEED, 99, len(rows)], **opts)
        rows.append((label, var_plain_atm / prices.var(ddof=1)))
    return rows


def convergence(path_counts, runs=32):
    ref = float(bs_price(S0, 100.0, T, R, SIGMA, Q, "call"))
    rmse = {m: [] for m in METHODS}
    for mi, method in enumerate(METHODS):
        for n in path_counts:
            prices, _, _ = _runs(method, "call", 100.0, n, runs, [SEED, 7, mi, n])
            rmse[method].append(np.sqrt(np.mean((prices - ref) ** 2)))
    return {m: np.array(v) for m, v in rmse.items()}


def plot(path_counts, rmse):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {
        "plain": "#2a78d6",
        "antithetic": "#eb6834",
        "control_variate": "#1baf7a",
        "sobol": "#eda100",
    }
    markers = {"plain": "o", "antithetic": "s", "control_variate": "^", "sobol": "D"}
    n = np.array(path_counts, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for method in METHODS:
        slope = np.polyfit(np.log(n), np.log(rmse[method]), 1)[0]
        label = f"{LABELS[method]}, slope {slope:.2f}".replace("-", "−")
        ax.plot(
            n,
            rmse[method],
            color=colors[method],
            lw=2,
            marker=markers[method],
            ms=5.5,
            mec=surface,
            mew=1.5,
            label=label,
        )

    # Direct labels at the line ends, nudged apart so close lines don't print on top of each other.
    min_gap = 0.16  # decades
    ends = sorted((np.log10(rmse[m][-1]), m) for m in METHODS)
    placed: list[float] = []
    for y, m in ends:
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(
            LABELS[m].split(" (")[0],
            (n[-1], rmse[m][-1]),
            xytext=(n[-1] * 1.25, 10**y_label),
            textcoords="data",
            va="center",
            color=ink,
            fontsize=8.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(right=n[-1] * 6)
    ax.set_xticks(n[::2])
    ax.set_xlabel("Paths (N)", color=ink2)
    ax.set_ylabel("RMSE vs Black–Scholes ($)", color=ink2)
    ax.set_title(
        "Same paths, smaller error: Sobol converges near 1/N, the rest at 1/√N",
        loc="left",
        color=ink,
        fontsize=11.5,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.035,
        f"ATM European call, S = K = {S0:.0f}, T = {T:.0f}y, r = {R:.0%}, "
        f"σ = {SIGMA:.0%} · 32 independent runs per point · seed {SEED}",
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
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="lower left")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true", help="skip the convergence figure")
    args = parser.parse_args()

    var, calib = vrf_and_calibration()
    print(
        f"Variance-reduction factor vs plain MC at {N:,} paths "
        f"({RUNS} independent runs per cell; sigma = {SIGMA:.0%}, T = {T:.0f}y)\n"
    )
    head = " | ".join(f"{kind} K={k:.0f}" for kind, k in CONTRACTS)
    print(f"| Method | {head} |")
    print("|---|" + "---:|" * len(CONTRACTS))
    for method in METHODS[1:]:
        cells = " | ".join(
            f"{var[(kind, k, 'plain')] / var[(kind, k, method)]:,.1f}×" for kind, k in CONTRACTS
        )
        print(f"| {LABELS[method]} | {cells} |")
    lo, hi = stats.f.ppf([0.025, 0.975], RUNS - 1, RUNS - 1)
    print(
        f"\nEach factor is a ratio of two {RUNS}-run variances: 95% CI is about "
        f"×{lo:.2f} to ×{hi:.2f} of the value shown."
    )

    print("\nError-bar calibration (all four contracts pooled):\n")
    print(
        "| Method | mean SE² / observed variance | share inside \\|z\\| < 1 | inside \\|z\\| < 2 |"
    )
    print("|---|---:|---:|---:|")
    passed = True
    for method in METHODS:
        z = np.concatenate(calib[method]["z"])
        ratios = np.array(calib[method]["se2"]) / np.array(calib[method]["var"])
        passed &= calibration_ok(z, ratios, RUNS)
        print(
            f"| {LABELS[method]} | {ratios.mean():.2f} | {np.mean(np.abs(z) < 1):.1%} "
            f"| {np.mean(np.abs(z) < 2):.1%} |"
        )
    n = len(CONTRACTS) * RUNS
    band1 = Z_BAND * np.sqrt(P_INSIDE_1 * (1 - P_INSIDE_1) / n)
    band2 = Z_BAND * np.sqrt(P_INSIDE_2 * (1 - P_INSIDE_2) / n)
    print(
        f"\nExpected: ratio 1, {P_INSIDE_1:.1%} ± {band1:.1%} and {P_INSIDE_2:.1%} ± {band2:.1%} "
        f"({n} runs per method, {Z_BAND:.0f}σ bands; Sobol's t-scores mapped to the normal "
        "scale)."
    )

    t = timings()
    print("\nEfficiency at equal wall-clock (ATM call, 262,144 paths, best of 5):\n")
    print("| Method | time (ms) | VRF | efficiency gain |")
    print("|---|---:|---:|---:|")
    for method in METHODS:
        vrf = var[("call", 100.0, "plain")] / var[("call", 100.0, method)]
        print(
            f"| {LABELS[method]} | {t[method] * 1e3:.1f} | {vrf:,.1f}× "
            f"| {vrf * t['plain'] / t[method]:,.1f}× |"
        )

    print(
        f"\nBrownian bridge: ATM call priced from {BRIDGE_STEPS}-step Sobol paths "
        f"({BRIDGE_RUNS} runs each, VRF vs plain MC):\n"
    )
    print("| Path construction | VRF |")
    print("|---|---:|")
    for label, vrf in bridge_table(var[("call", 100.0, "plain")]):
        print(f"| {label} | {vrf:,.1f}× |")

    if not args.no_figure:
        path_counts = [1 << p for p in range(8, 19)]
        plot(path_counts, convergence(path_counts))
        print(f"\nFigure written to {FIGURE}")

    print(f"\nCalibration gate: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
