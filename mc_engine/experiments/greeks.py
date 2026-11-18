"""Project 1 step 3: Monte Carlo Greeks against closed forms.

1. Gate: delta and vega from all three estimators vs Black–Scholes, on vanilla calls and puts
   and a digital call. Every valid estimate must sit within 4 standard errors (about 0.2%
   false-alarm chance across 28 checks). Pathwise on the digital must come back exactly 0:
   the gate checks that this documented failure stays visible rather than being masked.
2. Likelihood-ratio vs pathwise variance across maturities, in and at the money.
3. Figure: central-difference delta error vs bump size, with and without common random
   numbers, against the pathwise estimator.

    uv run python -m mc_engine.experiments.greeks
    uv run python -m mc_engine.experiments.greeks --no-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mc_engine import bs_delta, bs_vega, gbm
from mc_engine.black_scholes import bs_digital_call_delta, bs_digital_call_vega
from mc_engine.greeks import ESTIMATORS, mc_greeks
from mc_engine.pricer import payoff

S0, T, R, Q, SIGMA = 100.0, 1.0, 0.05, 0.01, 0.25
N = 1 << 18
SEED = 20261118
Z_TOL = 4.0
CONTRACTS = [
    ("call", 80.0),
    ("call", 100.0),
    ("call", 120.0),
    ("put", 100.0),
    ("digital_call", 100.0),
]
NAMES = {"pathwise": "Pathwise", "likelihood_ratio": "Likelihood ratio", "bump_crn": "Bump + CRN"}
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "greeks_bump_size.png"


def reference(kind, k, t=T):
    if kind == "digital_call":
        return (
            float(bs_digital_call_delta(S0, k, t, R, SIGMA, Q)),
            float(bs_digital_call_vega(S0, k, t, R, SIGMA, Q)),
        )
    return float(bs_delta(S0, k, t, R, SIGMA, Q, kind)), float(bs_vega(S0, k, t, R, SIGMA, Q))


def gate_table() -> bool:
    print(
        f"Delta and vega vs Black–Scholes: {N:,} paths, S0={S0:.0f}, sigma={SIGMA:.0%}, "
        f"T={T:.0f}y, r={R:.0%}, q={Q:.0%}. Cells: estimate (z-score).\n"
    )
    print("| Contract | Greek | Black–Scholes | " + " | ".join(NAMES.values()) + " |")
    print("|---|---|---:|" + "---:|" * len(ESTIMATORS))
    passed = True
    seeds = np.random.SeedSequence(SEED).spawn(len(CONTRACTS))
    for (kind, k), seed in zip(CONTRACTS, seeds, strict=True):
        res = mc_greeks(S0, k, T, R, SIGMA, Q, kind, N, seed)
        for greek, ref in zip(("delta", "vega"), reference(kind, k), strict=True):
            cells = []
            for est in ESTIMATORS:
                e = getattr(res, greek)[est]
                if kind == "digital_call" and est == "pathwise":
                    passed &= e.value == 0.0  # the failure mode must stay visible
                    cells.append(f"{e.value:.4f} ✗")
                    continue
                z = e.z_score(ref)
                passed &= abs(z) < Z_TOL
                cells.append(f"{e.value:.4f} ({z:+.1f})")
            label = "digital call" if kind == "digital_call" else kind
            print(f"| {label} K={k:.0f} | {greek} | {ref:.4f} | " + " | ".join(cells) + " |")
    print(
        "\n✗ = pathwise returns exactly 0 for the digital (its payoff derivative is 0 almost "
        "everywhere), with zero standard error."
    )
    return passed


def maturity_table() -> None:
    maturities = (2.0, 1.0, 0.25, 0.05, 0.01)
    print(
        f"\nVariance of likelihood ratio ÷ variance of pathwise (call, same paths, {N:,} each):\n"
    )
    print("| Strike | Greek | " + " | ".join(f"T={t}" for t in maturities) + " |")
    print("|---|---|" + "---:|" * len(maturities))
    for k in (80.0, 100.0):
        res = [
            mc_greeks(S0, k, t, R, SIGMA, Q, "call", N, [SEED, int(k), i])
            for i, t in enumerate(maturities)
        ]
        for greek in ("delta", "vega"):
            ratios = [
                (
                    getattr(g, greek)["likelihood_ratio"].std_error
                    / getattr(g, greek)["pathwise"].std_error
                )
                ** 2
                for g in res
            ]
            label = "80 (ITM)" if k == 80.0 else "100 (ATM)"
            print(f"| {label} | {greek} | " + " | ".join(f"{x:,.1f}×" for x in ratios) + " |")


def bump_study(n=1 << 15, runs=48):
    """RMSE of the central-difference delta vs bump size, ATM call."""
    bumps = np.geomspace(1e-3, 20.0, 13)
    ref = reference("call", 100.0)[0]
    disc = np.exp(-R * T)
    rmse = {"independent": [], "crn": []}
    rng = np.random.default_rng([SEED, 3])
    pathwise = []
    for h in bumps:
        err = {"independent": [], "crn": []}
        for _ in range(runs):
            z1, z2 = rng.standard_normal((2, n))
            s1 = gbm.terminal(S0, R, SIGMA, T, z1, q=Q)
            s2 = gbm.terminal(S0, R, SIGMA, T, z2, q=Q)
            up = payoff(s1 * (S0 + h) / S0, 100.0, "call")
            err["crn"].append(
                disc * np.mean(up - payoff(s1 * (S0 - h) / S0, 100.0, "call")) / (2 * h) - ref
            )
            err["independent"].append(
                disc * np.mean(up - payoff(s2 * (S0 - h) / S0, 100.0, "call")) / (2 * h) - ref
            )
            if h == bumps[0]:
                pathwise.append(disc * np.mean((s1 > 100.0) * s1 / S0) - ref)
        for key in rmse:
            rmse[key].append(np.sqrt(np.mean(np.square(err[key]))))
    return (
        bumps,
        {k: np.array(v) for k, v in rmse.items()},
        float(np.sqrt(np.mean(np.square(pathwise)))),
    )


def plot(bumps, rmse, pathwise_rmse) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(
        pathwise_rmse,
        color=ink2,
        lw=1.2,
        ls=(0, (4, 3)),
        zorder=1,
        label="Pathwise estimator (no bump)",
    )
    ax.plot(
        bumps,
        rmse["independent"],
        color=blue,
        lw=2,
        marker="o",
        ms=5.5,
        mec=surface,
        mew=1.5,
        zorder=3,
        label="Bump, independent draws up and down",
    )
    ax.plot(
        bumps,
        rmse["crn"],
        color=orange,
        lw=2,
        marker="s",
        ms=5,
        mec=surface,
        mew=1.5,
        zorder=3,
        label="Bump, common random numbers",
    )
    ax.annotate(
        "variance ∝ 1/h²",
        (bumps[1], rmse["independent"][1]),
        xytext=(12, 4),
        textcoords="offset points",
        color=ink,
        fontsize=8.5,
    )
    ax.annotate(
        "bias ∝ h²",
        (bumps[-1], rmse["crn"][-1]),
        xytext=(-56, 10),
        textcoords="offset points",
        color=ink,
        fontsize=8.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spot bump h ($, spot = 100)", color=ink2)
    ax.set_ylabel("RMSE of delta estimate", color=ink2)
    ax.set_title(
        "Common random numbers make the bump size almost free to choose",
        loc="left",
        color=ink,
        fontsize=11.5,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.035,
        f"Central-difference delta, ATM call, 32,768 paths, 48 runs per point · seed {SEED}",
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
    passed = gate_table()
    maturity_table()
    if not args.no_figure:
        bumps, rmse, pw = bump_study()
        plot(bumps, rmse, pw)
        print(
            f"\nBump study: pathwise RMSE {pw:.5f}; CRN RMSE from {rmse['crn'].min():.5f} to "
            f"{rmse['crn'].max():.5f}; independent RMSE from {rmse['independent'].min():.5f} "
            f"to {rmse['independent'].max():.2f}"
        )
        for h, a, b in zip(bumps, rmse["independent"], rmse["crn"], strict=True):
            print(f"  h={h:8.4f}  independent {a:10.5f}  crn {b:.5f}")
        print(f"Figure written to {FIGURE}")
    print(f"\nGreeks gate: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
