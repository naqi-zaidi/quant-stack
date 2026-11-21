"""Project 3 step 5, real-data half: the market maker on Nasdaq's AAPL day, walk-forward.

The day (2012-06-21, 9:30-16:00) is replayed message by message through the C++ book
(Project 2's LOBSTER replay), and makers trade against it with virtual orders that keep their
place in the queue (mm_strategy.replay). Nothing reacts to them: the synthetic half of step 5
(experiments.blind_spot) measures what that costs.

The day is cut into 30-minute blocks. Each block is a session: the maker starts flat, its
Avellaneda–Stoikov horizon is the block's end, and it is marked to the mid when the block
ends. 9:30-10:30 is the history block; the 11 blocks from 10:30 to 16:00 are traded
walk-forward. At the start of each, σ (60-second mid changes), k (the fill-intensity decay)
and the flow model are refitted on every earlier block. Nothing else is tuned on AAPL: γ = 3e-4
(the same risk aversion in dollars as on the synthetic market, where a tick and a lot are also
one cent and 100 shares), the 5-second horizon and the weight of 1 all come from steps 3 and 4.

Strategies: join the best bid and ask (the back of each queue), plain A–S, and A–S + flow.
Each also runs with new limit orders priced through its quotes counted as fills, as a
sensitivity (the default does not count them; see mm_strategy.replay).

The gate checks what must hold exactly: the replayed book matches LOBSTER's after every
message, the PnL identity holds for every maker in every block, and every block's flow model
refits exactly from the logs before it. Results are reported with standard errors across the
11 blocks; they come from one stock-day, and blocks of one day are not independent.

Needs the LOBSTER sample in data/ (see data/README.md), so it runs locally, not in CI.

    uv run python -m mm_strategy.experiments.aapl [--no-figure]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from marketdata import load_day
from mm_strategy.calibration import estimate_intensity, estimate_sigma
from mm_strategy.experiments.avellaneda import mean_se, paired
from mm_strategy.experiments.flow import GAMMA, HORIZON_S, WEIGHT
from mm_strategy.maker import AvellanedaStoikov, JoinBest, Quotes
from mm_strategy.replay import (
    CROSSED,
    HIDDEN,
    QUEUE,
    THROUGH,
    LobsterSource,
    ReplayMaker,
    TapeReplay,
    markouts,
)
from mm_strategy.signals import FEATURES, FlowFeatures, FlowSkewAS, fit, r2_oos, walk_forward

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "AAPL_2012-06-21_34200000_57600000_message_10.csv"
FIGURE = ROOT / "docs" / "figures" / "p3_aapl.png"
NS = 1_000_000_000
TICK = 100
DOLLAR = 10_000
OPEN = 34_200 * NS
BLOCK = 1_800 * NS
N_BLOCKS = 13
HISTORY = 2  # 9:30-10:30
DELTAS = np.arange(0.5, 20.01, 0.5)  # ticks; AAPL's spread is about 15 ticks wide
MARKOUT_S = (1, 2, 5, 10, 30)
JOIN, PLAIN, FLOW = "join the best", "A–S", "A–S + flow"
NAMES = (JOIN, PLAIN, FLOW)


class FeatureLogger:
    """Computes the flow features at every decision and never quotes."""

    def __init__(self) -> None:
        self.features = FlowFeatures()

    def quote(self, v) -> Quotes:
        self.features.update(v)
        return Quotes(None, None)


def block_bounds(i: int) -> tuple[int, int]:
    return OPEN + i * BLOCK, OPEN + (i + 1) * BLOCK


def clock(ns: int) -> str:
    s = ns // NS
    return f"{s // 3600}:{s % 3600 // 60:02d}"


class Day:
    """The LOBSTER files as arrays, cut into blocks that the calibration functions read as
    sessions (times from the block's start, executions grouped into aggressive orders by
    timestamp and side)."""

    def __init__(self, path: Path) -> None:
        day = load_day(str(path))
        m, b = day.messages, day.book
        self.ts = (m["time"].to_numpy() * NS).round().astype(np.int64)
        self.mid = (b["bid_price_1"].to_numpy(dtype=float)
                    + b["ask_price_1"].to_numpy(dtype=float)) / 2
        self.event = m["event"].to_numpy()
        self.side = m["direction"].to_numpy()
        self.price = m["price"].to_numpy()
        self.size = m["size"].to_numpy()

    def block(self, i: int) -> SimpleNamespace:
        start, end = block_bounds(i)
        ts, mid = self.ts - start, self.mid

        def mid_at(t):
            j = np.searchsorted(ts, np.asarray(t, dtype=np.int64), side="right") - 1
            return np.where(j >= 0, mid[np.maximum(j, 0)], np.nan)

        ex = (self.ts >= start) & (self.ts < end) & np.isin(self.event, (4, 5))
        ets, d = self.ts[ex], self.side[ex]
        new = np.r_[True, (ets[1:] != ets[:-1]) | (d[1:] != d[:-1])]
        trades = {"ts": ets - start, "price": self.price[ex], "qty": self.size[ex],
                  "taker_side": -d, "taker": np.cumsum(new)}
        return SimpleNamespace(params=SimpleNamespace(hours=BLOCK / NS / 3600, tick=TICK),
                               mid_at=mid_at, trades=trades)


def summarise(rm: ReplayMaker, quotes, end: int) -> dict:
    pnl = rm.pnl()
    f = rm.fill_arrays()
    q = np.array([x[1] for x in rm.samples]) if rm.samples else np.zeros(1)
    return {
        "net": pnl.net / DOLLAR, "spread": pnl.spread / DOLLAR,
        "inventory": pnl.inventory / DOLLAR, "rebates": -pnl.fees / DOLLAR,
        "volume": pnl.volume, "abs_q": float(np.abs(q).mean()), "identity": pnl.identity_holds,
        "kinds": {k: int(f["qty"][f["kind"] == k].sum()) for k in (QUEUE, THROUGH, HIDDEN,
                                                                  CROSSED)},
        "mk": markouts(f, quotes, end, TICK, MARKOUT_S)["all"],
    }


def col(rows, key) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def plot(blocks, results, r2, coef_path, fsd) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
    colors = {JOIN: orange, PLAIN: green, FLOW: blue}
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

    labels = [clock(block_bounds(i)[0]) for i in blocks]
    x = np.arange(len(blocks))

    ax = axes[0, 0]
    for k in NAMES:
        ax.plot(x, col(results[k], "net"), color=colors[k], marker="o", ms=3.5, lw=1.5, label=k)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    losses = int((col(results[PLAIN], "net") < 0).sum())
    swing = np.abs(col(results[JOIN], "net")).max()
    style(ax, f"A–S lost in {losses} of {len(blocks)} blocks; joining the best swung "
          f"up to ±${swing:,.0f}",
          "Block start (each block starts flat and is marked at its end)", "Net PnL ($ per block)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink)

    ax = axes[0, 1]
    ax.bar(x, np.array(r2) * 100, color=blue, width=0.6)
    m, se = mean_se(r2)
    ax.axhline(m * 100, color=ink, lw=1.2, ls="--")
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    style(ax, f"Walk-forward R² on AAPL: {m:.1%} ± {se:.1%} of 5 s moves",
          "Block start (model fitted on every earlier block)", "R² against no change (%)")

    ax = axes[1, 0]
    for k in NAMES:
        mk = np.array([[r["mk"][h] for h in MARKOUT_S] for r in results[k]])
        mu = np.nanmean(mk, axis=0)
        s_e = np.nanstd(mk, axis=0, ddof=1) / np.sqrt(np.isfinite(mk).sum(axis=0))
        ax.errorbar(MARKOUT_S, mu, yerr=1.96 * s_e, color=colors[k], marker="o", ms=3.5,
                    lw=1.5, capsize=2, label=k)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(MARKOUT_S)
    ax.set_xticklabels([f"{h:g}" for h in MARKOUT_S])
    ax.minorticks_off()
    mk10 = {k: np.nanmean([r["mk"][10] for r in results[k]]) for k in NAMES}
    style(ax, f"10 s markouts: {', '.join(f'{k} {v:+.1f}' for k, v in mk10.items())} ticks",
          "Seconds after the fill", "Markout, all fills (ticks/share), 95% CI")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink)

    ax = axes[1, 1]
    names = {"ofi": "order-flow imbalance, 5 s", "depth_imbalance": "depth imbalance, 3 ticks",
             "last_move": "mid change, last 1 s"}
    for j, (name, color) in enumerate(zip(FEATURES, (blue, orange, green), strict=True)):
        ax.plot(x, coef_path[:, j] * fsd[j], color=color, lw=1.6, marker="o", ms=3,
                label=names[name])
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    sign = "reversal" if coef_path[-1, 0] < 0 else "continuation"
    style(ax, f"On AAPL, past order flow predicts {sign}",
          "Block start (model fitted on every earlier block)",
          "Predicted 5 s move per 1 s.d. of the feature (ticks)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    if not DATA.exists():
        raise SystemExit(f"{DATA.relative_to(ROOT)} not found: see data/README.md")
    failures: list[str] = []
    t0 = time.time()
    day = Day(DATA)
    windows = [day.block(i) for i in range(N_BLOCKS)]
    rp = TapeReplay(LobsterSource(str(DATA)))

    # History block: features only.
    hist_logs = []
    for i in range(HISTORY):
        start, end = block_bounds(i)
        logger = FeatureLogger()
        rp.run([ReplayMaker(logger, start, end)], end)
        hist_logs.append(logger.features.log)

    test_blocks = list(range(HISTORY, N_BLOCKS))
    results = {k: [] for k in NAMES}
    crossing = {k: [] for k in NAMES}
    calib: list[tuple[float, float, float]] = []

    def trade(i, model):
        start, end = block_bounds(i)
        sigma = estimate_sigma(windows[:i], 60.0)
        fitk = estimate_intensity(windows[:i], DELTAS)
        calib.append((sigma.value, fitk.k.value, fitk.r2))
        base = AvellanedaStoikov(GAMMA, sigma.value, fitk.k.value)

        def strategies():
            return {JOIN: JoinBest(), PLAIN: base, FLOW: FlowSkewAS(base, model, WEIGHT)}

        makers = {k: ReplayMaker(s, start, end) for k, s in strategies().items()}
        cross = {k: ReplayMaker(s, start, end, cross_fills=True) for k, s in strategies().items()}
        logger = FeatureLogger()
        rp.run([*makers.values(), *cross.values(), ReplayMaker(logger, start, end)], end)
        q = rp.quotes()
        out = {k: summarise(rm, q, end) for k, rm in makers.items()}
        out_cross = {k: summarise(rm, q, end) for k, rm in cross.items()}
        log = logger.features.log
        return (out, out_cross, r2_oos(model, log), log), log

    wf = walk_forward(hist_logs, test_blocks, trade, HORIZON_S)
    r2 = []
    for _, _, (out, out_cross, r, _log) in wf:
        for k in NAMES:
            results[k].append(out[k])
            crossing[k].append(out_cross[k])
        r2.append(r)
    models = [m for _, m, _ in wf]
    coef_path = np.array([m.coef for m in models])

    # Exact checks.
    st = rp.rp.stats
    replay_ok = st.mismatches == 0 and st.errors == 0
    if not replay_ok:
        failures.append(f"replay diverged from LOBSTER: {st.mismatches} mismatches, "
                        f"{st.errors} errors")
    identity = all(r["identity"] for k in NAMES for r in results[k] + crossing[k])
    if not identity:
        failures.append("a PnL identity broke")
    block_logs = [res[3] for _, _, res in wf]
    clean = all(fit(hist_logs + block_logs[:j], HORIZON_S) == m and m.sessions == HISTORY + j
                for j, m in enumerate(models))
    if not clean:
        failures.append("a walk-forward model did not refit exactly from the earlier blocks")

    n = len(test_blocks)
    print(f"## AAPL 2012-06-21: {n} half-hour blocks, {clock(block_bounds(HISTORY)[0])}–16:00, "
          "walk-forward\n")
    print("| Check | Result | Gate |")
    print("|---|---|---|")
    print(f"| replayed book matches LOBSTER after every message | {st.events:,} messages, "
          f"{st.mismatches} mismatches, {st.errors} errors | {'yes' if replay_ok else '**NO**'} |")
    print(f"| PnL identity, every maker in every block | {'holds' if identity else 'broken'} | "
          f"{'yes' if identity else '**NO**'} |")
    print(f"| every block's flow model refits exactly from the blocks before it | "
          f"{'yes' if clean else 'no'} | {'yes' if clean else '**NO**'} |")

    s_vals = np.array(calib)
    print(f"\nWalk-forward inputs: σ {s_vals[:, 0].min():.2f}–{s_vals[:, 0].max():.2f} ticks/√s, "
          f"k {s_vals[:, 1].min():.3f}–{s_vals[:, 1].max():.3f} per tick (intensity fit R² "
          f"{s_vals[:, 2].min():.3f}–{s_vals[:, 2].max():.3f}); γ = {GAMMA:g}.")
    sp0 = AvellanedaStoikov(GAMMA, s_vals[0, 0], s_vals[0, 1])
    print(f"In the first test block, A–S quotes {sp0.total_spread(1800):.1f} ticks wide at the "
          f"block's start and {sp0.total_spread(0):.1f} at its end, and moves r "
          f"{-sp0.reservation_offset(1, 900):.1f} ticks per 100 shares at mid-block.")
    rm_, rse = mean_se(r2)
    print(f"Flow model R²_oos per block (%): mean {rm_ * 100:.2f} ± {rse * 100:.2f} "
          f"(t = {rm_ / rse:+.1f}); above zero in {int((np.array(r2) > 0).sum())} of {n}.")
    for j, name in enumerate(FEATURES):
        print(f"  coefficient on {name}: {coef_path[:, j].min():+.4f} to "
              f"{coef_path[:, j].max():+.4f}")

    print("\n| Strategy | Net PnL ($/hour) | Spread capture | Inventory PnL | Rebates | Volume "
          "(shares/hour) | Mean \\|inventory\\| | Markout 10 s (ticks) | Blocks > 0 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in NAMES:
        rows = results[k]
        net, nse = mean_se(2 * col(rows, "net"))
        mk = mean_se([r["mk"][10] for r in rows])
        print(f"| {k} | {net:+,.0f} ± {nse:,.0f} | {2 * col(rows, 'spread').mean():+,.0f} | "
              f"{2 * col(rows, 'inventory').mean():+,.0f} | "
              f"{2 * col(rows, 'rebates').mean():+,.0f} | "
              f"{2 * col(rows, 'volume').mean():,.0f} | {col(rows, 'abs_q').mean():,.0f} | "
              f"{mk[0]:+.2f} ± {mk[1]:.2f} | {int((col(rows, 'net') > 0).sum())} of {n} |")
    print("\n| Paired across blocks ($/hour) | Difference | t |")
    print("|---|---:|---:|")
    for a, b in ((FLOW, PLAIN), (PLAIN, JOIN), (FLOW, JOIN)):
        d, se = paired(2 * col(results[a], "net"), 2 * col(results[b], "net"))
        print(f"| {a} − {b} | {d:+,.0f} ± {se:,.0f} | {d / se:+.1f} |")

    print("\n| Strategy | Fills: queue ahead gone | traded through | hidden | "
          "With crossing orders as fills: net ($/hour) | crossed share of volume |")
    print("|---|---:|---:|---:|---:|---:|")
    for k in NAMES:
        kinds = {c: sum(r["kinds"][c] for r in results[k]) for c in (QUEUE, THROUGH, HIDDEN)}
        tot = max(sum(kinds.values()), 1)
        cr = crossing[k]
        cnet, cse = mean_se(2 * col(cr, "net"))
        csh = (sum(r["kinds"][CROSSED] for r in cr)
               / max(sum(sum(r["kinds"].values()) for r in cr), 1))
        print(f"| {k} | {kinds[QUEUE] / tot:.0%} | {kinds[THROUGH] / tot:.0%} | "
              f"{kinds[HIDDEN] / tot:.0%} | {cnet:+,.0f} ± {cse:,.0f} | {csh:.0%} |")

    if not args.no_figure:
        x = np.vstack([np.asarray(lg, float)[:, 2:] for lg in hist_logs])
        fsd = np.nanstd(x, axis=0)
        plot(test_blocks, results, r2, coef_path, fsd)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nRan in {time.time() - t0:.0f} s.")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
