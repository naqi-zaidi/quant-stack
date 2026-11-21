"""Project 3 step 4 gate: an order-flow signal in Avellaneda–Stoikov, fitted walk-forward.

The signal (mm_strategy.signals) predicts the mid's change over the next 5 seconds from three
features of everyone else's book: order-flow imbalance over the last 5 s, depth imbalance
within 3 ticks of the best, and the mid's change over the last second. "A–S + flow" is
Avellaneda–Stoikov (γ = 3e-4, step 3's choice) with its reservation price moved by that
prediction.

Protocol, in time order:
1. Calibration: σ and k from seeds 1001-1010 without a maker, exactly as in step 3.
2. History block: plain A–S trades seeds 1001-1010 and records its features. The horizon
   (5 s) and the weight on the prediction (1, so r is the expected mid) were fixed on this
   block before any test session was run; `--sensitivity` reproduces the evidence.
3. Test, walk-forward: seeds 1-N in order. Day d is traded with a model fitted on the history
   block and days 1..d-1 only, and its own log then joins the training set. Fixed spreads of
   1 and 3 ticks and plain A–S trade the same seeds, so every comparison is paired.

The gate requires:
- the walk-forward to be clean: day d's model refits exactly from the logs before day d;
- forecast skill out of sample: mean R²_oos over the test days above 0 at 4 standard errors
  (against the no-change forecast);
- the PnL identity to hold exactly in every session;
- A–S + flow's net PnL above plain A–S's, at 4 paired standard errors. On the history block,
  cross-fitted, the difference was t = 6.7 over 10 sessions, so 10 sessions is enough for this.
Everything else, including the comparisons with fixed spreads, is reported, not gated. The
README's table comes from --sessions 40; CI runs --sessions 10.

    uv run python -m mm_strategy.experiments.flow [--sessions 10] [--no-figure] [--sensitivity]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mm_strategy.accounting import decompose, inventory_stats, maker_markouts
from mm_strategy.calibration import estimate_intensity, estimate_sigma
from mm_strategy.experiments.avellaneda import CAL_SEEDS, mean_se, paired
from mm_strategy.maker import AvellanedaStoikov, FixedSpread, MarketMaker
from mm_strategy.market import INFORMED, NOISE_MARKET, MarketParams, simulate
from mm_strategy.signals import (
    FEATURES,
    FlowSkewAS,
    fit,
    leave_one_out_r2,
    r2_oos,
    training_rows,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_flow.png"
HISTORY_SEEDS = CAL_SEEDS  # 1001-1010
GAMMA = 3e-4
HORIZON_S = 5.0
WEIGHT = 1.0
R2_HORIZONS = (1, 2, 5, 10, 20, 30)
MARKOUT_S = (1, 2, 5, 10, 30)
DOLLAR = 10_000
FLOW = "A–S + flow"
PLAIN = "A–S"


def run_session(p: MarketParams, seed: int, strategy) -> dict:
    mm = MarketMaker(strategy)
    s = simulate(p, seed, participant=mm)
    pnl = decompose(s, mm)
    if not pnl.identity_holds:
        raise AssertionError(f"PnL identity broke for seed {seed}")
    f = mm.fill_arrays()
    mk = maker_markouts(s, MARKOUT_S)
    q = np.array([x[1] for x in mm.samples])
    return {
        "net": pnl.net / DOLLAR, "spread": pnl.spread / DOLLAR,
        "inventory": pnl.inventory / DOLLAR, "rebates": -pnl.fees / DOLLAR,
        "volume": pnl.volume, "abs_q": float(np.abs(q).mean()),
        "half_life": inventory_stats(s, mm)["half_life_s"],
        "drawdown": inventory_stats(s, mm)["max_drawdown"] / DOLLAR,
        "informed": float(f["qty"][f["counterparty"] == INFORMED].sum()
                          / max(f["qty"].sum(), 1)),
        "mk_informed": mk.get(INFORMED, dict.fromkeys(MARKOUT_S, np.nan)),
        "mk_noise": mk.get(NOISE_MARKET, dict.fromkeys(MARKOUT_S, np.nan)),
        "mk_all": mk["all"],
    }


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def history_block(p, base) -> tuple[list, list[dict]]:
    """Plain A–S on the history seeds, recording its features."""
    logs, rows = [], []
    for sd in HISTORY_SEEDS:
        st = FlowSkewAS(base)
        rows.append(run_session(p, sd, st))
        logs.append(st.features.log)
    return logs, rows


def feature_sd(logs) -> np.ndarray:
    x = np.vstack([training_rows(log, HORIZON_S)[0] for log in logs])
    return x.std(axis=0)


def sensitivity(p, base, logs, plain_rows) -> None:
    """Cross-fitted on the history block: session i is traded with a model fitted on the
    other nine. This is the evidence the horizon and weight were fixed on."""
    plain = column(plain_rows, "net")
    print("\n## Sensitivity on the history block (cross-fitted; not the test sessions)\n")
    print("| Horizon | Weight | A–S + flow − A–S ($/hour) | t |")
    print("|---:|---:|---:|---:|")
    for h in (2.0, 5.0, 10.0, 20.0):
        for w in (0.5, 1.0, 2.0):
            nets = [run_session(p, sd, FlowSkewAS(base, fit(logs[:i] + logs[i + 1:], h), w))["net"]
                    for i, sd in enumerate(HISTORY_SEEDS)]
            d, se = paired(nets, plain)
            print(f"| {h:g} s | {w:g} | {d:+,.0f} ± {se:,.0f} | {d / se:+.1f} |", flush=True)


def plot(r2_by_h, test_r2, coef_path, fsd, results, n) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
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

    # Forecast skill by horizon (history, leave one session out) and on the test days.
    ax = axes[0, 0]
    hs = np.array(R2_HORIZONS, float)
    m = np.array([r2_by_h[h].mean() for h in R2_HORIZONS]) * 100
    se = np.array([r2_by_h[h].std(ddof=1) / np.sqrt(len(r2_by_h[h])) for h in R2_HORIZONS]) * 100
    ax.errorbar(hs, m, yerr=1.96 * se, color=blue, marker="o", ms=4, lw=1.6, capsize=2,
                label="history block, leave one session out")
    tm, tse = mean_se(test_r2)
    ax.errorbar([HORIZON_S], [tm * 100], yerr=[1.96 * tse * 100], color=orange, marker="D",
                ms=6, capsize=3, lw=1.4, label=f"test days, walk-forward (n = {n})")
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{h:g}" for h in hs])
    ax.minorticks_off()
    style(ax, f"Out of sample the signal explains {tm:.1%} of 5 s mid moves",
          "Forecast horizon (s)", "R² against the no-change forecast (%), 95% CI")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper right")

    # Walk-forward coefficients, scaled to ticks per standard deviation of each feature.
    ax = axes[0, 1]
    days = np.arange(1, len(coef_path) + 1)
    names = {"ofi": "order-flow imbalance, 5 s", "depth_imbalance": "depth imbalance, 3 ticks",
             "last_move": "mid change, last 1 s"}
    for j, (name, color) in enumerate(zip(FEATURES, (blue, orange, green), strict=True)):
        ax.plot(days, coef_path[:, j] * fsd[j], color=color, lw=1.6, marker="o", ms=2.5,
                label=names[name])
    ax.axhline(0, color=ink2, lw=0.8)
    style(ax, "Refitted daily, the model barely moves",
          "Test day (model fitted on the history block and every earlier day)",
          "Predicted 5 s move per 1 s.d. of the feature (ticks)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="center right")

    # Markouts by counterparty: colour is the strategy, line style the counterparty.
    ax = axes[1, 0]
    gain, handles = {}, []
    for group, ls, who in (("mk_noise", "--", "noise"), ("mk_informed", "-", "informed")):
        means = {}
        for key, color in ((PLAIN, green), (FLOW, blue)):
            rows = results[key]
            mk = np.array([[r[group][h] for h in MARKOUT_S] for r in rows])
            mu = np.nanmean(mk, axis=0)
            s_e = np.nanstd(mk, axis=0, ddof=1) / np.sqrt(np.isfinite(mk).sum(axis=0))
            ax.errorbar(MARKOUT_S, mu, yerr=1.96 * s_e, color=color, ls=ls, marker="o", ms=3.5,
                        lw=1.5, capsize=2)
            handles.append(Line2D([], [], color=color, ls=ls, lw=1.5,
                                  label=f"{key}, fills vs {who}"))
            means[key] = mu[MARKOUT_S.index(10)]
        gain[group] = means[FLOW] - means[PLAIN]
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(MARKOUT_S)
    ax.set_xticklabels([f"{h:g}" for h in MARKOUT_S])
    ax.minorticks_off()
    style(ax, f"10 s markouts gain {gain['mk_informed']:+.2f} vs informed, "
          f"{gain['mk_noise']:+.2f} vs noise",
          "Seconds after the fill", "Markout (ticks per share), 95% CI")
    ax.legend(handles=handles, frameon=False, fontsize=7, labelcolor=ink, loc="center right",
              handlelength=2.6)

    # Paired session PnL.
    ax = axes[1, 1]
    a, b = column(results[PLAIN], "net"), column(results[FLOW], "net")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pad = 0.06 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color=ink2, lw=0.9, ls="--")
    ax.axhline(0, color=grid, lw=0.8)
    ax.axvline(0, color=grid, lw=0.8)
    ax.scatter(a, b, s=16, color=blue, zorder=3)
    wins = int((b > a).sum())
    ax.text(0.97, 0.04, "above the dashed line: A–S + flow made more", transform=ax.transAxes,
            fontsize=7.5, color=ink2, ha="right", va="bottom")
    style(ax, f"A–S + flow beat plain A–S in {wins} of {len(a)} test sessions",
          "Plain A–S net PnL ($/hour)", "A–S + flow net PnL ($/hour)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also rerun the history-block sweep over horizon and weight (~4 min)")
    args = ap.parse_args(argv)
    p = MarketParams()
    failures: list[str] = []
    t0 = time.time()

    # 1. Calibration, as in step 3.
    cal = [simulate(p, sd) for sd in CAL_SEEDS]
    sigma = estimate_sigma(cal, 60.0)
    k = estimate_intensity(cal).k
    base = AvellanedaStoikov(GAMMA, sigma.value, k.value)

    # 2. History block.
    hist_logs, hist_rows = history_block(p, base)
    r2_by_h = {h: leave_one_out_r2(hist_logs, h) for h in R2_HORIZONS}
    fsd = feature_sd(hist_logs)
    print(f"## History block: plain A–S (γ = {GAMMA:g}, σ = {sigma.value:.3f}, "
          f"k = {k.value:.3f}) on seeds 1001–1010\n")
    print("Forecast skill, each session predicted by a model fitted on the other nine:\n")
    print("| Horizon | R²_oos (%) |")
    print("|---:|---:|")
    for h in R2_HORIZONS:
        r = r2_by_h[h]
        print(f"| {h} s | {r.mean() * 100:.2f} ± {r.std(ddof=1) / np.sqrt(len(r)) * 100:.2f} |")
    print(f"\nWhich features carry the {HORIZON_S:g} s forecast "
          "(R²_oos %, leave one session out):\n")
    print("| Features | R²_oos (%) |")
    print("|---|---:|")
    for cols in ((0,), (1,), (2,), (0, 1), (0, 1, 2)):
        r = leave_one_out_r2(hist_logs, HORIZON_S, columns=cols)
        print(f"| {' + '.join(FEATURES[c] for c in cols)} | "
              f"{r.mean() * 100:.2f} ± {r.std(ddof=1) / np.sqrt(len(r)) * 100:.2f} |")
    if args.sensitivity:
        sensitivity(p, base, hist_logs, hist_rows)

    # 3. Test sessions: baselines, then the walk-forward.
    n = args.sessions
    seeds = list(range(1, n + 1))
    results: dict[str, list[dict]] = {
        "fixed 1 tick": [run_session(p, sd, FixedSpread(1)) for sd in seeds],
        "fixed 3 ticks": [run_session(p, sd, FixedSpread(3)) for sd in seeds],
        PLAIN: [run_session(p, sd, FlowSkewAS(base)) for sd in seeds],
    }

    def trade(seed, model):
        st = FlowSkewAS(base, model, WEIGHT)
        r = run_session(p, seed, st)
        r["r2"] = r2_oos(model, st.features.log)
        r["shift_sd"] = float(np.std(st.shifts))
        r["log"] = np.asarray(st.features.log, dtype=float)
        return r, r["log"]

    wf = walk_forward(hist_logs, seeds, trade, HORIZON_S)
    results[FLOW] = [r for _, _, r in wf]
    models = [m for _, m, _ in wf]
    coef_path = np.array([m.coef for m in models])

    # Walk-forward integrity: rebuild every day's model from the logs before it.
    test_logs = [r["log"] for r in results[FLOW]]
    clean = all(fit(hist_logs + test_logs[:i], HORIZON_S) == m
                and m.sessions == len(HISTORY_SEEDS) + i for i, m in enumerate(models))
    if not clean:
        failures.append("a walk-forward model did not refit exactly from the logs before it")

    test_r2 = column(results[FLOW], "r2")
    r2m, r2se = mean_se(test_r2)
    print(f"\n## Test: {n} one-hour sessions, seeds 1–{n}, walk-forward, paired\n")
    print(f"Horizon {HORIZON_S:g} s, weight {WEIGHT:g}. Day 1's model was fitted on "
          f"{models[0].rows:,} decisions from the 10 history sessions; day {n}'s on "
          f"{models[-1].rows:,} from {models[-1].sessions}.\n")
    print("| Walk-forward check | Result | Gate |")
    print("|---|---|---|")
    print(f"| every day's model refits exactly from the logs before that day | "
          f"{'yes' if clean else 'no'} | {'yes' if clean else '**NO**'} |")
    ok = r2m / r2se > 4
    print(f"| R²_oos on the test days, against no change (%) | {r2m * 100:.2f} ± "
          f"{r2se * 100:.2f} (t = {r2m / r2se:+.1f}); every day > 0: "
          f"{'yes' if (test_r2 > 0).all() else 'no'} | > 0 at 4σ: {'yes' if ok else '**NO**'} |")
    if not ok:
        failures.append(f"no out-of-sample forecast skill (t = {r2m / r2se:+.1f})")
    lo, hi = coef_path.min(axis=0), coef_path.max(axis=0)
    for j, name in enumerate(FEATURES):
        print(f"| coefficient on {name}, range over the {n} refits | {lo[j]:+.4f} to "
              f"{hi[j]:+.4f} | reported |")
    print(f"| reservation-price shift, s.d. (ticks) | "
          f"{np.mean(column(results[FLOW], 'shift_sd')):.2f} | reported |")

    print("\n| Strategy | Net PnL ($/hour) | Spread capture | Inventory PnL | Rebates | Volume | "
          "Mean \\|inventory\\| | PnL s.d. | Traded with informed | Markout vs informed, "
          "10 s (ticks) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, rows in results.items():
        net, nse = mean_se(column(rows, "net"))
        mk = mean_se([r["mk_informed"][10] for r in rows])
        print(f"| {name} | {net:+,.0f} ± {nse:,.0f} | {column(rows, 'spread').mean():+,.0f} | "
              f"{column(rows, 'inventory').mean():+,.0f} | "
              f"{column(rows, 'rebates').mean():+,.0f} | "
              f"{column(rows, 'volume').mean():,.0f} | {column(rows, 'abs_q').mean():,.0f} | "
              f"{np.std(column(rows, 'net'), ddof=1):,.0f} | "
              f"{column(rows, 'informed').mean():.0%} | {mk[0]:+.2f} ± {mk[1]:.2f} |")

    print("\n## Paired comparisons (difference ± paired s.e.)\n")
    print("| Comparison | Difference | t | Gate |")
    print("|---|---:|---:|---|")
    flow = results[FLOW]
    for key, label in (("net", "net PnL ($)"), ("spread", "spread capture ($)"),
                       ("inventory", "inventory PnL ($)"), ("abs_q", "mean |inventory| (shares)")):
        d, se = paired(column(flow, key), column(results[PLAIN], key))
        gate = "reported"
        if key == "net":
            ok = d / se > 4
            gate = f"> 0 at 4σ: {'yes' if ok else '**NO**'}"
            if not ok:
                failures.append(f"A–S + flow did not beat plain A–S at 4σ (t = {d / se:+.1f})")
        print(f"| {label.replace('|', chr(92) + '|')}, {FLOW} − {PLAIN} | {d:+,.0f} ± {se:,.0f} | "
              f"{d / se:+.1f} | {gate} |")
    for other in ("fixed 1 tick", "fixed 3 ticks"):
        d, se = paired(column(flow, "net"), column(results[other], "net"))
        print(f"| net PnL ($), {FLOW} − {other} | {d:+,.0f} ± {se:,.0f} | {d / se:+.1f} | "
              "reported |")
    net, nse = mean_se(column(flow, "net"))
    print(f"\n{FLOW} net PnL: {net:+,.0f} ± {nse:,.0f} $/hour (t = {net / nse:+.1f} against zero), "
          "maker rebates included.")

    if not args.no_figure:
        plot(r2_by_h, test_r2, coef_path, fsd, results, n)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nRan in {time.time() - t0:.0f} s.")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
