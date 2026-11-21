"""Project 3 step 3 gate: Avellaneda–Stoikov, calibrated to the market, against a fixed spread.

1. Calibration, on sessions of their own (seeds 1001-1010, no maker):
   - sigma from 60-second mid changes must land within 4 standard errors of the fair value's
     true volatility. The signature plot shows the bid-ask bounce inflating shorter intervals.
   - Fill intensity must decay exponentially with distance from the mid: R² of the log-linear
     fit at least 0.95. Its slope is k.
2. Evaluation on seeds 1-N (default 10; the step 2 baseline used the same seeds, so every
   comparison is paired: same fair value, same noise and informed arrival times). The strategies
   are fixed spreads of 1 and 3 ticks, Avellaneda–Stoikov at gamma = 0 (its spread 2/k without
   the inventory skew, a control), and five values of gamma > 0. The gate requires the PnL
   identity to hold exactly in every session, and that the skew controls inventory: at every
   gamma > 0, mean |inventory| must be below the gamma = 0 control's, at 4 paired standard
   errors.

PnL comparisons and the PnL-risk frontier are reported with their paired standard errors, not
gated. A fixed spread carries about 1,000 shares through a random walk, so its PnL is noisy
(s.d. ~$1,500 an hour), and separating it from Avellaneda–Stoikov at 4 standard errors takes
more sessions than CI should spend. The README's table comes from --sessions 40; CI runs
--sessions 6.

    uv run python -m mm_strategy.experiments.avellaneda [--sessions 10] [--no-figure]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mm_strategy.accounting import decompose, inventory_stats, maker_markouts
from mm_strategy.calibration import (
    estimate_intensity,
    estimate_sigma,
    signature,
    true_sigma,
)
from mm_strategy.maker import AvellanedaStoikov, FixedSpread, MarketMaker
from mm_strategy.market import INFORMED, MarketParams, simulate

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_avellaneda.png"
CAL_SEEDS = range(1001, 1011)
GAMMAS = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
CHOSEN = 3e-4
SIG_STEPS = (1, 2, 5, 10, 20, 30, 60, 120, 300)
DOLLAR = 10_000


def paired(a, b) -> tuple[float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def mean_se(xs) -> tuple[float, float]:
    a = np.asarray(xs, float)
    a = a[np.isfinite(a)]
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def evaluate(p: MarketParams, strategy, seeds) -> dict[str, list]:
    out: dict[str, list] = {k: [] for k in ("net", "spread", "inventory", "rebates", "volume",
                                            "abs_q", "half_life", "drawdown", "informed",
                                            "mk10", "path")}
    for sd in seeds:
        mm = MarketMaker(strategy)
        s = simulate(p, sd, participant=mm)
        pnl = decompose(s, mm)
        if not pnl.identity_holds:
            raise AssertionError(f"PnL identity broke for seed {sd}")
        inv = inventory_stats(s, mm)
        q = np.array([x[1] for x in mm.samples])
        f = mm.fill_arrays()
        out["net"].append(pnl.net / DOLLAR)
        out["spread"].append(pnl.spread / DOLLAR)
        out["inventory"].append(pnl.inventory / DOLLAR)
        out["rebates"].append(-pnl.fees / DOLLAR)
        out["volume"].append(pnl.volume)
        out["abs_q"].append(float(np.abs(q).mean()))
        out["half_life"].append(inv["half_life_s"])
        out["drawdown"].append(inv["max_drawdown"] / DOLLAR)
        out["informed"].append(float(f["qty"][f["counterparty"] == INFORMED].sum()
                                     / max(f["qty"].sum(), 1)))
        out["mk10"].append(maker_markouts(s, (10,))["all"][10])
        if sd == seeds[0]:
            out["path"] = [(t, qq) for t, qq, _ in mm.samples]
    return out


def plot(sig, true_sig, fit, results, labels) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, green, yellow = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), dpi=150)
    fig.patch.set_facecolor(surface)

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(surface)
        ax.set_title(title, loc="left", color=ink, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, color=ink2, fontsize=8.5)
        ax.set_ylabel(ylabel, color=ink2, fontsize=8.5)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)

    ax = axes[0, 0]
    hs = np.array(list(sig))
    m = np.array([sig[h].value for h in hs])
    se = np.array([sig[h].se for h in hs])
    ax.errorbar(hs, m, yerr=1.96 * se, color=blue, marker="o", ms=4, lw=1.6, capsize=2)
    ax.axhline(true_sig, color=ink, lw=1.2, ls="--")
    ax.text(hs[0], true_sig * 0.93, f"fair value's true σ = {true_sig:.2f}", color=ink,
            fontsize=7.5, va="top")
    ax.set_xscale("log")
    style(ax, "Short intervals see the bid-ask bounce, not volatility",
          "Sampling interval (s)", "Mid volatility (ticks per √s), 95% CI")

    ax = axes[0, 1]
    ax.semilogy(fit.deltas, fit.curve, "o", color=blue, ms=4, label="measured")
    xx = np.linspace(fit.deltas[0], fit.deltas[-1], 50)
    ax.semilogy(xx, fit.A.value * np.exp(-fit.k.value * xx), color=orange, lw=1.6,
                label=f"A·exp(−kδ): k = {fit.k.value:.3f} ± {fit.k.se:.3f} per tick")
    ticks = [0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    style(ax, f"Fill intensity decays exponentially (R² {fit.r2:.3f})",
          "Distance from the mid, δ (ticks)", "Orders reaching δ, per second per side")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink)

    ax = axes[1, 0]
    for key, color in (("fixed 1 tick", orange), (f"A–S γ={CHOSEN:g}", green)):
        path = np.array(results[key]["path"])
        ax.plot(path[:, 0] / 60e9, path[:, 1], color=color, lw=1.0, label=key)
    ax.axhline(0, color=ink2, lw=0.8)
    style(ax, "The inventory skew keeps the position near flat",
          "Minutes (session 1)", "Inventory (shares)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="lower left")

    ax = axes[1, 1]
    colors = {"fixed": orange, "control": yellow, "as": green}
    for key in labels:
        r = results[key]
        risk = float(np.std(r["net"], ddof=1))
        mean, se = mean_se(r["net"])
        kind = "fixed" if key.startswith("fixed") else ("control" if key == "A–S γ=0" else "as")
        ax.errorbar(risk, mean, yerr=1.96 * se, color=colors[kind], marker="o", ms=5,
                    capsize=2, lw=1)
        label = key.replace("A–S ", "").replace("γ=0", "γ=0 (no skew)", 1) if kind == "control" \
            else key.replace("A–S ", "")
        # Low-risk points crowd together: alternate their labels above and below.
        dy = -11 if key.endswith(("0.001", "3e-05")) else 4
        ax.annotate(label, (risk, mean), xytext=(6, dy), textcoords="offset points",
                    fontsize=7, color=ink2)
    ax.axhline(0, color=ink2, lw=0.8)
    best_fixed = max((k for k in labels if k.startswith("fixed")),
                     key=lambda k: np.mean(results[k]["net"]))
    ratio = (np.std(results[best_fixed]["net"], ddof=1)
             / np.std(results[f"A–S γ={CHOSEN:g}"]["net"], ddof=1))
    style(ax, f"γ={CHOSEN:g}: the best fixed spread's PnL at 1/{ratio:.0f} of its risk",
          "PnL standard deviation across sessions ($/hour)", "Mean net PnL ($/hour), 95% CI")

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    p = MarketParams()
    failures: list[str] = []

    # 1. Calibration.
    cal = [simulate(p, sd) for sd in CAL_SEEDS]
    sig = signature(cal, SIG_STEPS)
    sigma = estimate_sigma(cal, 60.0)
    true_sig = true_sigma(p)
    fit = estimate_intensity(cal)
    z = (sigma.value - true_sig) / sigma.se
    print("## Calibration (seeds 1001–1010, no maker)\n")
    print("| Input | Estimate ± s.e. | Check |")
    print("|---|---|---|")
    print(f"| σ, 60 s mid changes (ticks per √s) | {sigma.value:.3f} ± {sigma.se:.3f} | true "
          f"{true_sig:.3f}: {z:+.1f} s.e. ({'within' if abs(z) < 4 else '**outside**'} 4) |")
    print(f"| σ, 1 s mid changes | {sig[1].value:.3f} ± {sig[1].se:.3f} | bid-ask bounce: "
          f"{sig[1].value / true_sig:.1f}× the true value |")
    print(f"| k, fill-intensity decay (per tick) | {fit.k.value:.3f} ± {fit.k.se:.3f} | "
          f"log-linear R² {fit.r2:.3f} ({'≥' if fit.r2 >= 0.95 else '**<**'} 0.95) |")
    print(f"| A, intensity at the mid (per s, per side) | {fit.A.value:.3f} ± {fit.A.se:.3f} | – |")
    if abs(z) >= 4:
        failures.append(f"sigma estimate {z:+.1f} s.e. from the true value")
    if fit.r2 < 0.95:
        failures.append(f"fill intensity is not exponential (R² {fit.r2:.3f})")

    # 2. Evaluation.
    seeds = list(range(1, args.sessions + 1))
    strategies = {
        "fixed 1 tick": FixedSpread(1),
        "fixed 3 ticks": FixedSpread(3),
        "A–S γ=0": AvellanedaStoikov(0.0, sigma.value, fit.k.value),
    }
    for g in GAMMAS:
        strategies[f"A–S γ={g:g}"] = AvellanedaStoikov(g, sigma.value, fit.k.value)
    results = {name: evaluate(p, st, seeds) for name, st in strategies.items()}

    n = args.sessions
    sp0 = AvellanedaStoikov(CHOSEN, sigma.value, fit.k.value)
    print(f"\n## Evaluation: {n} one-hour sessions, seeds 1–{n}, paired\n")
    print(f"Avellaneda–Stoikov uses σ = {sigma.value:.3f} ticks/√s and k = {fit.k.value:.3f}/tick "
          f"from the calibration; at the open, γ = {CHOSEN:g} quotes a total spread of "
          f"{sp0.total_spread(3600):.2f} ticks and moves its reservation price "
          f"{-sp0.reservation_offset(1, 3600):.2f} ticks per 100 shares held.\n")
    print("| Strategy | Net PnL ($/hour) | Spread capture | Inventory PnL | Volume | "
          "Mean \\|inventory\\| | Half-life (s) | PnL s.d. | Traded with informed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        net, nse = mean_se(r["net"])
        hl, _ = mean_se(r["half_life"])
        print(f"| {name} | {net:+,.0f} ± {nse:,.0f} | {np.mean(r['spread']):+,.0f} | "
              f"{np.mean(r['inventory']):+,.0f} | {np.mean(r['volume']):,.0f} | "
              f"{np.mean(r['abs_q']):,.0f} | {hl:,.0f} | {np.std(r['net'], ddof=1):,.0f} | "
              f"{np.mean(r['informed']):.0%} |")

    print("\n## Paired comparisons (difference ± paired s.e.)\n")
    print("| Comparison | Difference | t | Gate |")
    print("|---|---:|---:|---|")
    control = results["A–S γ=0"]
    for g in GAMMAS:
        d, se = paired(results[f"A–S γ={g:g}"]["abs_q"], control["abs_q"])
        ok = d / se < -4
        print(f"| mean \\|inventory\\|, γ={g:g} − γ=0 (shares) | {d:+,.0f} ± {se:,.0f} | "
              f"{d / se:+.1f} | < 0 at 4σ: {'yes' if ok else '**NO**'} |")
        if not ok:
            failures.append(f"gamma {g:g} did not reduce inventory at 4σ (t = {d / se:+.1f})")
    chosen = results[f"A–S γ={CHOSEN:g}"]
    for other in ("fixed 1 tick", "fixed 3 ticks", "A–S γ=0"):
        d, se = paired(chosen["net"], results[other]["net"])
        print(f"| net PnL, γ={CHOSEN:g} − {other} ($) | {d:+,.0f} ± {se:,.0f} | {d / se:+.1f} | "
              "reported |")

    if not args.no_figure:
        plot(sig, true_sig, fit, results, list(results))
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
