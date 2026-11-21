"""Project 3 step 5 gate, synthetic half: how far a replay is from trading live.

A replay keeps the recorded market fixed, so nobody reacts to the maker: no informed trader
picks off its stale quote, and no noise trader prices off it. On the synthetic market the same
strategy can trade both ways, live and on a recording of the same session, which measures
what that blind spot costs. It is measured here before the replay is trusted on AAPL.

1. Calibration and history block as in step 4: σ and k from seeds 1001-1010 without a maker;
   plain A–S on the same seeds records features, and the flow model is fitted on them once.
2. For each seed 1-N, one session with no maker is recorded as a LOBSTER tape (a silent maker
   records what a strategy would have seen at every decision). Each strategy then trades that
   seed live, in its own session, and replayed on the tape (mm_strategy.replay).

The gate requires:
- fidelity: with no maker, the replay shows a strategy exactly the market the live session
  showed, all five levels of each side, at every decision of every session;
- the PnL identity for every replayed maker.
Everything else is reported: each strategy's replay bias (replay − live, paired by seed),
whether the differences between strategies survive the replay, and what counting new limit
orders priced through the maker's quote as fills would add. The README's tables come from
--sessions 40; CI runs --sessions 6.

    uv run python -m mm_strategy.experiments.blind_spot [--sessions 6] [--no-figure]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mm_strategy.accounting import decompose, maker_markouts
from mm_strategy.calibration import estimate_intensity, estimate_sigma
from mm_strategy.experiments.avellaneda import CAL_SEEDS, mean_se, paired
from mm_strategy.experiments.flow import GAMMA, HORIZON_S, WEIGHT, history_block
from mm_strategy.maker import AvellanedaStoikov, FixedSpread, MarketMaker, Quotes
from mm_strategy.market import INFORMED, MarketParams, simulate
from mm_strategy.replay import (
    CROSSED,
    QUEUE,
    THROUGH,
    ReplayMaker,
    TapeReplay,
    TapeSource,
    markouts,
)
from mm_strategy.signals import FlowSkewAS, fit

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_blind_spot.png"
DOLLAR = 10_000
MARKOUT_S = (1, 2, 5, 10, 30)
FLOW, PLAIN = "A–S + flow", "A–S"


class Recorder:
    """A strategy that never quotes and remembers what it saw."""

    def __init__(self) -> None:
        self.views: list = []

    def quote(self, v) -> Quotes:
        self.views.append((v.now, v.bids, v.asks))
        return Quotes(None, None)


def summarise(pnl, f: dict, mk: dict) -> dict:
    qty = f["qty"]
    return {
        "net": pnl.net / DOLLAR, "spread": pnl.spread / DOLLAR,
        "inventory": pnl.inventory / DOLLAR, "volume": pnl.volume,
        "informed": float(qty[f["counterparty"] == INFORMED].sum() / max(qty.sum(), 1)),
        "mk_informed": mk.get(INFORMED, dict.fromkeys(MARKOUT_S, np.nan)),
        "mk_all": mk["all"],
        "kinds": {k: int(qty[f["kind"] == k].sum()) for k in (QUEUE, THROUGH, CROSSED)}
        if "kind" in f else {},
    }


def live(p, seed, strategy) -> dict:
    mm = MarketMaker(strategy)
    s = simulate(p, seed, participant=mm)
    pnl = decompose(s, mm)
    if not pnl.identity_holds:
        raise AssertionError(f"live PnL identity broke for seed {seed}")
    return summarise(pnl, mm.fill_arrays(), maker_markouts(s, MARKOUT_S))


def col(rows, key) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def plot(results, names, n) -> None:
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

    # Net PnL, live vs replayed, per strategy.
    ax = axes[0, 0]
    y = np.arange(len(names))[::-1]
    for dy, mode, color, label in ((0.22, "live", ink, "live"),
                                   (0.0, "replay", blue, "replayed"),
                                   (-0.22, "cross", orange, "replayed, crossing orders fill")):
        m = np.array([mean_se(col(results[k][mode], "net")) for k in names])
        ax.errorbar(m[:, 0], y + dy, xerr=1.96 * m[:, 1], fmt="o", color=color, ms=4.5,
                    capsize=2, lw=1.2, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(0, color=ink2, lw=0.8)
    bias = [paired(col(results[k]["replay"], "net"), col(results[k]["live"], "net"))[0]
            for k in names]
    style(ax, f"Replayed PnL is off by {min(bias):+,.0f} to {max(bias):+,.0f} \\$ an hour",
          "Net PnL ($/hour), 95% CI", "")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="lower left")

    # The flow signal's gain, session by session.
    ax = axes[0, 1]
    a = col(results[FLOW]["live"], "net") - col(results[PLAIN]["live"], "net")
    b = col(results[FLOW]["replay"], "net") - col(results[PLAIN]["replay"], "net")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color=ink2, lw=0.9, ls="--")
    ax.scatter(a, b, s=16, color=blue, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    style(ax, f"The flow signal's gain: {a.mean():+,.0f} \\$/hour live, {b.mean():+,.0f} replayed",
          "A–S + flow − A–S, live ($/hour)", "A–S + flow − A–S, replayed ($/hour)")

    # Markouts against informed traders, live vs replayed (A–S).
    ax = axes[1, 0]
    for mode, color in (("live", ink), ("replay", blue)):
        mk = np.array([[r["mk_informed"][h] for h in MARKOUT_S] for r in results[PLAIN][mode]])
        mu = np.nanmean(mk, axis=0)
        s_e = np.nanstd(mk, axis=0, ddof=1) / np.sqrt(np.isfinite(mk).sum(axis=0))
        ax.errorbar(MARKOUT_S, mu, yerr=1.96 * s_e, color=color, marker="o", ms=3.5, lw=1.5,
                    capsize=2)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(MARKOUT_S)
    ax.set_xticklabels([f"{h:g}" for h in MARKOUT_S])
    ax.minorticks_off()
    sh_l = col(results[PLAIN]["live"], "informed").mean()
    sh_r = col(results[PLAIN]["replay"], "informed").mean()
    style(ax, f"Replayed, A–S trades less with informed flow ({sh_r:.0%} vs {sh_l:.0%})",
          "Seconds after the fill", "A–S markout vs informed (ticks/share), 95% CI")
    ax.legend(handles=[Line2D([], [], color=ink, lw=1.5, label="live"),
                       Line2D([], [], color=blue, lw=1.5, label="replayed")],
              frameon=False, fontsize=7.5, labelcolor=ink, loc="upper right")

    # Replayed volume by how the fill came about, with crossing orders allowed to fill.
    ax = axes[1, 1]
    x = np.arange(len(names))
    parts = ((QUEUE, "at its price, queue ahead gone", green),
             (THROUGH, "traded through its price", blue),
             (CROSSED, "crossed by a new limit order", orange))
    bottom = np.zeros(len(names))
    for kind, label, color in parts:
        v = np.array([np.mean([r["kinds"][kind] for r in results[k]["cross"]]) for k in names])
        ax.bar(x, v, bottom=bottom, color=color, width=0.6, label=label)
        bottom += v
    lv = np.array([col(results[k]["live"], "volume").mean() for k in names])
    ax.scatter(x, lv, marker="_", s=500, color=ink, lw=2, zorder=3, label="live volume")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    cross_share = np.mean([np.mean([r["kinds"][CROSSED] / max(sum(r["kinds"].values()), 1)
                                    for r in results[k]["cross"]]) for k in names])
    style(ax, f"Crossing orders: {cross_share:.0%} of replayed volume, none live",
          "", "Shares filled per hour (replayed, crossing orders fill)")
    ax.legend(frameon=False, fontsize=7, labelcolor=ink, loc="upper right")

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    p = MarketParams()
    failures: list[str] = []
    t0 = time.time()

    cal = [simulate(p, sd) for sd in CAL_SEEDS]
    base = AvellanedaStoikov(GAMMA, estimate_sigma(cal, 60.0).value,
                             estimate_intensity(cal).k.value)
    model = fit(history_block(p, base)[0], HORIZON_S)
    strategies = {
        "fixed 1 tick": lambda: FixedSpread(1),
        "fixed 3 ticks": lambda: FixedSpread(3),
        PLAIN: lambda: base,
        FLOW: lambda: FlowSkewAS(base, model, WEIGHT),
    }
    names = list(strategies)
    results = {k: {"live": [], "replay": [], "cross": []} for k in names}
    faithful = 0
    seeds = list(range(1, args.sessions + 1))
    for sd in seeds:
        rec_live = Recorder()
        s = simulate(p, sd, participant=MarketMaker(rec_live), record_tape=True)
        for k, make in strategies.items():
            results[k]["live"].append(live(p, sd, make()))
        rec = Recorder()
        makers = {k: ReplayMaker(make(), 0, p.duration_ns) for k, make in strategies.items()}
        crossing = {k: ReplayMaker(make(), 0, p.duration_ns, cross_fills=True)
                    for k, make in strategies.items()}
        rp = TapeReplay(TapeSource(s.tape))
        rp.run([ReplayMaker(rec, 0, p.duration_ns), *makers.values(), *crossing.values()],
               p.duration_ns)
        faithful += rec.views == rec_live.views
        q = rp.quotes()
        for mode, group in (("replay", makers), ("cross", crossing)):
            for k, rm in group.items():
                pnl = rm.pnl()
                if not pnl.identity_holds:
                    failures.append(f"replayed PnL identity broke: {k}, seed {sd}")
                f = rm.fill_arrays()
                results[k][mode].append(summarise(pnl, f, markouts(f, q, p.duration_ns, p.tick,
                                                                   MARKOUT_S)))

    n = len(seeds)
    print(f"## Live vs replayed: {n} one-hour sessions, seeds 1–{n}\n")
    ok = faithful == n
    print("| Check | Result | Gate |")
    print("|---|---|---|")
    print(f"| with no maker, the replay shows exactly the live market at every decision "
          f"(5 levels a side, 36,000 decisions a session) | {faithful} of {n} sessions | "
          f"{'yes' if ok else '**NO**'} |")
    print(f"| PnL identity, every replayed maker | "
          f"{'holds' if not failures else 'broken'} | {'yes' if not failures else '**NO**'} |")
    if not ok:
        failures.append(f"replay differed from the live market in {n - faithful} sessions")

    print("\n| Strategy | Live net ($/hour) | Replayed net | Replay bias | "
          "Replayed, crossing orders fill | Their bias |")
    print("|---|---:|---:|---:|---:|---:|")
    for k in names:
        lv, rp_, cr = (col(results[k][m], "net") for m in ("live", "replay", "cross"))
        b, bse = paired(rp_, lv)
        c, cse = paired(cr, lv)
        print(f"| {k} | {lv.mean():+,.0f} ± {mean_se(lv)[1]:,.0f} | "
              f"{rp_.mean():+,.0f} ± {mean_se(rp_)[1]:,.0f} | {b:+,.0f} ± {bse:,.0f} | "
              f"{cr.mean():+,.0f} ± {mean_se(cr)[1]:,.0f} | {c:+,.0f} ± {cse:,.0f} |")

    print("\n| Strategy | Volume, live | Volume, replayed | Informed share, live | replayed | "
          "Markout vs informed 10 s, live | replayed | Spread capture, live | replayed |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in names:
        L, R = results[k]["live"], results[k]["replay"]
        mkl = np.nanmean([r["mk_informed"][10] for r in L])
        mkr = np.nanmean([r["mk_informed"][10] for r in R])
        print(f"| {k} | {col(L, 'volume').mean():,.0f} | {col(R, 'volume').mean():,.0f} | "
              f"{col(L, 'informed').mean():.0%} | {col(R, 'informed').mean():.0%} | "
              f"{mkl:+.2f} | {mkr:+.2f} | {col(L, 'spread').mean():+,.0f} | "
              f"{col(R, 'spread').mean():+,.0f} |")

    print("\n## Do the differences between strategies survive the replay?\n")
    print("| Difference (net, $/hour) | Live | Replayed | Replayed − live |")
    print("|---|---:|---:|---:|")
    for x, y in ((FLOW, PLAIN), (PLAIN, "fixed 3 ticks"), (PLAIN, "fixed 1 tick"),
                 ("fixed 3 ticks", "fixed 1 tick")):
        dl = col(results[x]["live"], "net") - col(results[y]["live"], "net")
        dr = col(results[x]["replay"], "net") - col(results[y]["replay"], "net")
        print(f"| {x} − {y} | {dl.mean():+,.0f} ± {mean_se(dl)[1]:,.0f} | "
              f"{dr.mean():+,.0f} ± {mean_se(dr)[1]:,.0f} | "
              f"{(dr - dl).mean():+,.0f} ± {mean_se(dr - dl)[1]:,.0f} |")

    if not args.no_figure:
        plot(results, names, n)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nRan in {time.time() - t0:.0f} s.")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
