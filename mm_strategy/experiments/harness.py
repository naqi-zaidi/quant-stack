"""Project 3 step 2 gate: the market-making harness keeps exact books.

A fixed-spread maker quotes `half_spread` ticks either side of the market mid (1, 2 and 3 ticks)
in `--sessions` seeded one-hour sessions each. The gate checks:

1. Reconciliation. In every session, the fills, inventory, cash and fees the maker booked as
   fills arrived equal those recomputed from the engine's own trade record.
2. The PnL identity, net = spread capture + inventory PnL - fees, holds exactly (integer price
   units) in every session.
3. The maker never takes liquidity (its quotes are post-only) and never trades with itself.
4. Reproducible, and fair to the other traders. The same seed and strategy give the same
   session. Adding the maker changes neither the fair value nor when the noise traders act: the
   count of their engine calls equals that of the same seed without the maker.

Reported, not gated: the baseline's PnL split, markouts by counterparty, inventory and drawdown.

    uv run python -m mm_strategy.experiments.harness [--sessions 12] [--no-figure]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mm_strategy.accounting import (
    decompose,
    inventory_stats,
    maker_fills_from_engine,
    maker_markouts,
    mid2_before,
    reconcile,
)
from mm_strategy.maker import FixedSpread, MarketMaker
from mm_strategy.market import INFORMED, NOISE_MARKET, NS, MarketParams, Session, simulate

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_harness.png"
HALF_SPREADS = (1, 2, 3)
HORIZONS = (0.001, 1, 10, 60)
DOLLAR = 10_000


def run(p: MarketParams, seed: int, half_spread: float) -> tuple[Session, MarketMaker]:
    mm = MarketMaker(FixedSpread(half_spread=half_spread))
    return simulate(p, seed, participant=mm), mm


def mean_se(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def pnl_paths(s: Session, mm: MarketMaker) -> dict[str, np.ndarray]:
    """Cumulative spread capture, fees, net PnL (marked to mid) and inventory, each second."""
    f = maker_fills_from_engine(s)
    m2 = mid2_before(s, f["ts"])
    spread_i = f["side"] * f["qty"] * (m2 / 2 - f["price"])
    fee_i = np.where(f["passive"], -mm.fees.maker_rebate, mm.fees.taker_fee) * f["qty"]
    smp = np.array(mm.samples, dtype=np.int64)
    step = max(1, int(round(1000 / mm.requote_ms)))
    ts, inv, cash = smp[::step, 0], smp[::step, 1], smp[::step, 2]
    k = np.searchsorted(f["ts"], ts, side="right")
    spread_cum = np.concatenate([[0], np.cumsum(spread_i)])[k]
    fee_cum = np.concatenate([[0], np.cumsum(fee_i)])[k]
    net = cash + inv * s.mid_at(ts)
    return {"t": ts / NS / 60, "spread": spread_cum / DOLLAR, "fees": fee_cum / DOLLAR,
            "net": net / DOLLAR, "inventory_pnl": (net - spread_cum + fee_cum) / DOLLAR,
            "inventory": inv}


def plot(first: tuple[Session, MarketMaker], table: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=150,
                             gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
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

    s, mm = first
    path = pnl_paths(s, mm)
    ax = axes[0]
    ax.plot(path["t"], path["spread"], color=green, lw=1.6, label="spread capture")
    ax.plot(path["t"], path["inventory_pnl"], color=orange, lw=1.6, label="inventory PnL")
    ax.plot(path["t"], path["net"], color=ink, lw=1.8, label="net (after rebates)")
    ax.axhline(0, color=ink2, lw=0.8)
    style(ax, "Spread earned, then lost to the inventory",
          "Minutes (session 1, 1-tick half-spread)", "Cumulative PnL ($)")
    for key, color in (("spread", green), ("inventory_pnl", orange), ("net", ink)):
        ax.text(path["t"][-1] + 0.5, path[key][-1], f"{path[key][-1]:+,.0f}", color=color,
                fontsize=7.5, va="center")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="lower left")

    ax = axes[1]
    ax.plot(path["t"], path["inventory"], color=blue, lw=1.0)
    ax.axhline(2000, color=ink2, lw=0.8, ls="--")
    ax.axhline(-2000, color=ink2, lw=0.8, ls="--")
    style(ax, "Inventory swings between the ±2,000 limits", "Minutes (session 1)", "Shares")

    ax = axes[2]
    x = np.arange(len(HALF_SPREADS))
    w = 0.26
    for j, (key, label, color) in enumerate((("spread", "spread capture", green),
                                             ("inventory", "inventory PnL", orange),
                                             ("net", "net", ink))):
        m = [table[h][key][0] for h in HALF_SPREADS]
        se = [table[h][key][1] for h in HALF_SPREADS]
        ax.bar(x + (j - 1) * w, m, w, yerr=[1.96 * e for e in se], color=color, label=label,
               error_kw={"ecolor": ink2, "lw": 0.8, "capsize": 2})
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} tick{'s' if h > 1 else ''}" for h in HALF_SPREADS])
    style(ax, "Inventory losses exceed spread capture", "Half-spread quoted",
          "PnL per one-hour session ($), 95% CI")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    p = MarketParams()
    seeds = range(1, args.sessions + 1)
    failures: list[str] = []

    plain = {sd: simulate(p, sd) for sd in seeds}
    table: dict = {}
    first = None
    for h in HALF_SPREADS:
        rows = []
        for sd in seeds:
            s, mm = run(p, sd, h)
            if first is None:
                first = (s, mm)
            tag = f"half-spread {h}, seed {sd}"
            for prob in reconcile(s, mm):
                failures.append(f"{tag}: {prob}")
            pnl = decompose(s, mm)
            if not pnl.identity_holds:
                failures.append(f"{tag}: PnL identity does not hold")
            fills = maker_fills_from_engine(s)
            if not fills["passive"].all():
                failures.append(f"{tag}: the maker took liquidity")
            if not np.array_equal(s.fair_value, plain[sd].fair_value):
                failures.append(f"{tag}: the fair value changed")
            for who in ("noise limit", "noise market"):
                if s.calls[who] != plain[sd].calls[who]:
                    failures.append(f"{tag}: {who} calls {s.calls[who]} vs "
                                    f"{plain[sd].calls[who]} without the maker")
            mk = maker_markouts(s, HORIZONS)
            inv = inventory_stats(s, mm)
            vol = pnl.volume
            informed = int(fills["qty"][fills["counterparty"] == INFORMED].sum())
            rows.append({
                "net": pnl.net / DOLLAR, "spread": pnl.spread / DOLLAR,
                "inventory": pnl.inventory / DOLLAR, "rebates": -pnl.fees / DOLLAR,
                "volume": vol, "per_share": pnl.net / vol / p.tick if vol else np.nan,
                "informed_share": informed / vol if vol else np.nan,
                "mk_all": mk["all"], "mk_noise": mk.get(NOISE_MARKET, {}),
                "mk_informed": mk.get(INFORMED, {}),
                "max_inv": inv["max_abs_inventory"], "half_life": inv["half_life_s"],
                "drawdown": inv["max_drawdown"] / DOLLAR,
            })
        table[h] = {k: mean_se([r[k] for r in rows]) for k in
                    ("net", "spread", "inventory", "rebates", "volume", "per_share",
                     "informed_share", "max_inv", "half_life", "drawdown")}
        for grp in ("mk_all", "mk_noise", "mk_informed"):
            table[h][grp] = {hz: mean_se([r[grp].get(hz, np.nan) for r in rows])
                             for hz in HORIZONS}

    again_s, again_mm = run(p, 1, HALF_SPREADS[0])
    if again_s.digest != first[0].digest or reconcile(again_s, first[1]):
        failures.append("the same seed and strategy did not reproduce the session")

    n = args.sessions
    print(f"## Fixed-spread maker: {n} one-hour sessions per half-spread, seeds 1–{n}\n")
    print("Quotes 100 shares a side, post-only, refreshed every 100 ms, inventory limit ±2,000 "
          "shares; rebate $0.0020 a share. PnL marked to the book's mid at the close.\n")
    print("| Half-spread | Net PnL ($) | Spread capture ($) | Inventory PnL ($) | Rebates ($) | "
          "Volume (shares) | Net per share (ticks) | Traded with informed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for h in HALF_SPREADS:
        t = table[h]

        def f(k, fmt="{:+,.0f}", t=t):
            return f"{fmt.format(t[k][0])} ± {fmt.format(t[k][1]).lstrip('+')}"

        print(f"| {h} tick{'s' if h > 1 else ''} | {f('net')} | {f('spread')} | "
              f"{f('inventory')} | {f('rebates')} | {f('volume', '{:,.0f}')} | "
              f"{f('per_share', '{:+.2f}')} | {t['informed_share'][0]:.0%} |")
    print("\n## The maker's markouts (ticks per share)\n")
    print("| Half-spread | Against | At the fill | 1 s | 10 s | 60 s |")
    print("|---|---|---:|---:|---:|---:|")
    for h in HALF_SPREADS:
        for grp, label in (("mk_noise", "noise takers"), ("mk_informed", "informed takers"),
                           ("mk_all", "all")):
            cells = " | ".join(f"{table[h][grp][hz][0]:+.2f}" for hz in HORIZONS)
            print(f"| {h} | {label} | {cells} |")
    print("\n## Inventory\n")
    print("| Half-spread | Max \\|inventory\\| (shares) | Half-life (s) | Max drawdown ($) |")
    print("|---|---:|---:|---:|")
    for h in HALF_SPREADS:
        t = table[h]
        print(f"| {h} | {t['max_inv'][0]:,.0f} | {t['half_life'][0]:,.0f} ± "
              f"{t['half_life'][1]:,.0f} | {t['drawdown'][0]:,.0f} ± {t['drawdown'][1]:,.0f} |")

    print(f"\nChecked in all {n * len(HALF_SPREADS)} sessions: books reconcile with the engine's "
          "trades; the PnL identity holds exactly; no liquidity taken; fair value and noise "
          "traders' calls unchanged by the maker. Seed 1 reproduces.")
    if not args.no_figure:
        plot(first, table)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures[:20]:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
