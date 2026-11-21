"""Project 3 step 1 gate: the synthetic market behaves like one a market maker must survive.

Over `--sessions` seeded one-hour sessions (default 20):

1. Reproducible. Seed 1 run twice gives the same book event digest and the same trades; seed 2
   gives different ones.
2. Two-sided. Both sides are quoted at least 99% of the time in every session.
3. Prices follow the hidden fair value. In every session the median |mid - V| is under 5 ticks,
   and 60-second mid moves correlate with fair-value moves (mean over sessions at least 0.6).
4. Adverse selection exists, at 4 standard errors across sessions (10-second markouts, ticks
   per share): informed takers gain, noise takers lose, passive orders lose to informed flow
   and gain from noise flow.

Also reported, not gated: spread, depth, and how much of the 10-second price moves order-flow
imbalance explains, set against the AAPL day of Project 2 step 6.

    uv run python -m mm_strategy.experiments.market [--sessions 20] [--no-figure]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mm_strategy.market import NS, MarketParams, Session, simulate
from mm_strategy.markouts import by_role
from orderbook import Side
from orderbook.analytics import NO_ASK, NO_BID
from orderbook.microstructure import ofi_regression

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_market.png"
HORIZONS = (0.001, 1, 2, 5, 10, 20, 30, 60)
ROLES = [
    (("informed", "taker"), "informed takers"),
    (("noise market", "taker"), "noise takers"),
    (("noise limit", "maker vs noise market"), "passive vs noise flow"),
    (("noise limit", "maker vs informed"), "passive vs informed flow"),
]
AAPL_OFI_R2 = 0.41  # Project 2 step 6, 10-second intervals


def session_stats(s: Session) -> dict:
    p = s.params
    end = p.duration_ns
    two, one = s.spread.two_sided_ns(end), s.spread.one_sided_ns(end)
    grid = np.arange(1, int(p.hours * 3600)) * NS
    err = (s.mid_at(grid) - s.fair_value_at(grid)) / p.tick
    g60 = grid[::60]
    rm, rv = np.diff(s.mid_at(g60)), np.diff(s.fair_value_at(g60))
    ok = np.isfinite(rm)
    reg = ofi_regression(s.quotes, 0, end, bucket_s=10.0, boot=0) if len(s.quotes["e"]) else None
    return {
        "two_sided": two / (two + one),
        "spread": s.spread.mean(end),
        "depth_best": float(np.mean([s.depth.mean(sd, end)[0] for sd in Side])),
        "track_median": float(np.nanmedian(np.abs(err))),
        "corr60": float(np.corrcoef(rm[ok], rv[ok])[0, 1]),
        "ofi_r2": reg.r2 if reg else float("nan"),
        "trades": len(s.trades["ts"]),
        "markouts": by_role(s, HORIZONS),
    }


def mean_se(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def plot(first: Session, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150,
                             gridspec_kw={"width_ratios": [1.3, 1]})
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

    # Ten minutes of the first session: quotes around the hidden fair value.
    ax = axes[0]
    p = first.params
    t0, t1 = 20 * 60 * NS, 25 * 60 * NS
    q = first.quotes
    ts = q["ts"].astype(np.int64)
    sel = (ts >= t0) & (ts <= t1)
    tt = (ts[sel] - t0) / NS / 60
    ask = np.where(q["ask"][sel] == NO_ASK, np.nan, q["ask"][sel] / 1e4)
    bid = np.where(q["bid"][sel] == NO_BID, np.nan, q["bid"][sel] / 1e4)
    ax.step(tt, ask, where="post", color=colors[1], lw=0.9, label="best ask")
    ax.step(tt, bid, where="post", color=colors[0], lw=0.9, label="best bid")
    fv_t = np.arange(t0, t1, p.fv_step_ms * 1_000_000 * 10)
    ax.plot((fv_t - t0) / NS / 60, first.fair_value_at(fv_t) / 1e4, color=ink, lw=1.4,
            label="hidden fair value")
    tr = first.trades
    tts = tr["ts"].astype(np.int64)
    inf = ((tr["taker"].astype(np.int64) >> 40) == 3) & (tts >= t0) & (tts <= t1)
    ax.scatter((tts[inf] - t0) / NS / 60, tr["price"][inf] / 1e4, s=8, color=colors[3], zorder=3,
               label="informed trades")
    style(ax, "Informed traders pull the quotes back to the fair value",
          "Minutes (session 1, 20:00 to 25:00)", "Price ($)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper left", ncol=2)

    # Markout curves.
    ax = axes[1]
    hs = np.array([0.0, *HORIZONS[1:]])  # the first horizon (1 ms) is drawn at the fill
    for (key, label), c in zip(ROLES, colors, strict=True):
        vals = np.array([[r["markouts"][key][h] for h in HORIZONS] for r in rows])
        m, se = vals.mean(0), vals.std(0, ddof=1) / np.sqrt(len(vals))
        ax.plot(hs, m, color=c, lw=2, marker="o", ms=3.5, mec=surface, label=label)
        ax.fill_between(hs, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.18, lw=0)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 10, 60])
    ax.set_xticklabels(["fill", "1 s", "10 s", "60 s"])
    ax.set_xlim(0, 66)
    style(ax, "Passive orders lose to informed flow", "Time after the fill",
          "Markout (ticks per share), 95% CI")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    p = MarketParams()
    failures: list[str] = []

    # 1. Reproducible.
    a, b, c = simulate(p, 1), simulate(p, 1), simulate(p, 2)
    same = a.digest == b.digest and all(np.array_equal(a.trades[k], b.trades[k]) for k in a.trades)
    differs = a.digest != c.digest
    if not (same and differs):
        failures.append(f"reproducibility: same seed equal={same}, other seed differs={differs}")

    sessions = [a] + [simulate(p, seed) for seed in range(2, args.sessions + 1)]
    rows = [session_stats(s) for s in sessions]

    print(f"## Synthetic market: {args.sessions} one-hour sessions, seeds 1–{args.sessions}\n")
    print(f"Fair value: GBM, {p.sigma:.0%} annualised, ${p.s0:.0f} at the open. Noise limit "
          f"orders {p.limit_rate:g}/s (mean life {p.mean_life_s:g} s), noise market orders "
          f"{p.market_rate:g}/s, informed sweeps {p.informed_rate:g}/s at an edge of "
          f"{p.edge_ticks} ticks. Tick one cent.\n")
    print("| Statistic | Mean ± s.e. over sessions | Worst session | Gate |")
    print("|---|---|---|---|")

    def line(name, key, worst_fn, gate, fmt="{:.3f}"):
        vals = [r[key] for r in rows]
        m, se = mean_se(vals)
        worst = worst_fn(vals)
        print(f"| {name} | {fmt.format(m)} ± {fmt.format(se)} | {fmt.format(worst)} | {gate} |")
        return m, worst

    _, worst = line("Share of time quoted on both sides", "two_sided", min,
                    "≥ 0.99 in every session", "{:.4f}")
    if worst < 0.99:
        failures.append(f"a session was two-sided only {worst:.4f} of the time")
    _, worst = line("Median \\|mid − fair value\\| (ticks)", "track_median", max,
                    "< 5 in every session", "{:.2f}")
    if worst >= 5:
        failures.append(f"a session's mid strayed a median {worst:.2f} ticks from fair value")
    m, _ = line("Correlation of 60 s mid and fair-value moves", "corr60", min, "mean ≥ 0.6")
    if m < 0.6:
        failures.append(f"60 s moves correlate only {m:.3f} with the fair value")
    line("Mean spread (ticks, time-weighted)", "spread", max, "–", "{:.2f}")
    line("Mean depth at the best (shares)", "depth_best", min, "–", "{:.0f}")
    m, _ = line("OFI R², 10 s intervals", "ofi_r2", min, f"– (AAPL day: {AAPL_OFI_R2})")
    line("Trades per session", "trades", min, "–", "{:,.0f}")

    print("\n## Adverse selection: 10-second markouts, ticks per share\n")
    print("| Who | Volume per session (shares) | Markout at the fill | 10 s markout ± s.e. | "
          "t | Gate |")
    print("|---|---:|---:|---:|---:|---|")
    expected = {"informed takers": 1, "noise takers": -1, "passive vs noise flow": 1,
                "passive vs informed flow": -1}
    for key, label in ROLES:
        m0, _ = mean_se([r["markouts"][key][HORIZONS[0]] for r in rows])
        m, se = mean_se([r["markouts"][key][10] for r in rows])
        vol, _ = mean_se([r["markouts"][key]["volume"] for r in rows])
        t = m / se
        sign = expected[label]
        ok = sign * t > 4
        print(f"| {label} | {vol:,.0f} | {m0:+.2f} | {m:+.2f} ± {se:.2f} | {t:+.1f} | "
              f"{'> 0' if sign > 0 else '< 0'} at 4σ: {'yes' if ok else '**NO**'} |")
        if not ok:
            failures.append(f"{label}: 10 s markout t = {t:+.1f}, expected sign {sign:+d} at 4σ")

    print(f"\nReproducible: seed 1 twice gives digest {a.digest:#018x} both times; seed 2 gives "
          f"{c.digest:#018x}.")
    if not args.no_figure:
        plot(a, rows)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for f in failures:
        print(f"  - {f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
