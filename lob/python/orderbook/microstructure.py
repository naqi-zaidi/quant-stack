"""Phase 2 step 6 gate: microstructure analytics from the event stream match brute force.

1. Synthetic flows (runs in CI). Five fixed-seed regimes (orderbook.gate.flow) go through both
   books with all four analytics attached. After every call, each value is recomputed here in
   Python from the book's own views, and must match exactly:
   - the time-weighted spread distribution and depth sums;
   - the rebuilt L3 book;
   - every resting order's queue position (every 25 calls) and the tracked histories;
   - OFI: each quote update's e_n, recomputed from the recorded quotes; the last-N window;
     the total.
   Map and array books must give identical analytics.
2. Real day (local only, needs a LOBSTER day in data/). The AAPL day is replayed with the
   analytics attached, and each one is recomputed straight from LOBSTER's orderbook file, which
   is Nasdaq's book, independent of the engine. The time-weighted spread distribution, the depth
   profile at 0-9 ticks and the cumulative OFI must match exactly. Queue positions of sampled
   orders are checked against the replayed book's L3 view.
3. Figure (local; skipped with --no-figure): spread distribution, depth profile, OFI against
   mid-price change in 10-second intervals, and queue positions over time.

    uv run python -m orderbook.microstructure
    uv run python -m orderbook.microstructure --no-figure
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orderbook import ArrayOrderBook, OrderBook, Side
from orderbook import lobster as lb
from orderbook.analytics import (
    NO_ASK,
    NO_BID,
    DepthProfile,
    OrderFlowImbalance,
    QueuePosition,
    QueuePositions,
    QueueStatus,
    SpreadDistribution,
)
from orderbook.gate import REGIMES, find_lobster_day, flow

ROOT = Path(__file__).resolve().parents[3]
FIGURE = ROOT / "docs" / "figures" / "aapl_microstructure.png"
BOOKS = {"map": OrderBook, "array": ArrayOrderBook}

SYN_TICKS, SYN_WINDOW = 8, 16
DAY_TICKS, DAY_WINDOW = 10, 100
CENT = 100  # LOBSTER price units per cent: the day's tick


# ---------------------------------------------------------------------------------------------
# OFI, written out as in Cont, Kukanov and Stoikov (2014), eq. for e_n. Quotes are
# (bid, bid_qty, ask, ask_qty); an empty side is (NO_BID, 0) or (NO_ASK, 0), which orders the
# same as -infinity and +infinity.


def ofi_e(prev: tuple, cur: tuple) -> int:
    pb0, qb0, pa0, qa0 = prev
    pb1, qb1, pa1, qa1 = cur
    return (
        (qb1 if pb1 >= pb0 else 0)
        - (qb0 if pb1 <= pb0 else 0)
        - (qa1 if pa1 <= pa0 else 0)
        + (qa0 if pa1 >= pa0 else 0)
    )


EMPTY_QUOTE = (NO_BID, 0, NO_ASK, 0)


def quote_tuple(q) -> tuple:
    """An analytics Quote as (bid, bid_qty, ask, ask_qty) with the empty-side sentinels."""
    return (
        NO_BID if q.bid is None else q.bid,
        q.bid_qty,
        NO_ASK if q.ask is None else q.ask,
        q.ask_qty,
    )


# ---------------------------------------------------------------------------------------------
# 1. Synthetic flows.


@dataclass
class SynReport:
    regime: str
    calls: int = 0
    quote_updates: int = 0
    positions: int = 0
    samples: int = 0
    problems: list[str] = field(default_factory=list)

    def problem(self, what: str) -> None:
        if len(self.problems) < 5:
            self.problems.append(what)


def run_synthetic(regime, seed: int, calls: int, book_cls) -> tuple[SynReport, dict]:
    rep = SynReport(regime.name)
    spread, depth = SpreadDistribution(1), DepthProfile(1, SYN_TICKS)
    ofi, queue = OrderFlowImbalance(SYN_WINDOW, record=True), QueuePositions()
    book = book_cls(tick_size=1, expected_orders=4096, record_events=False)
    for listener in (spread, depth, ofi, queue):
        book.attach(listener)

    spread_bf: dict[int, int] = defaultdict(int)
    one_sided_bf = 0
    depth_bf = {s: [0] * SYN_TICKS for s in Side}
    observed_bf = dict.fromkeys(Side, 0)
    last: dict[int, tuple] = {}  # tracked id -> (orders ahead, qty ahead, leaves)
    history_bf: list[tuple] = []

    for call in flow(regime, seed, calls):
        ts = call[-1]
        if call[0] == "submit":
            _, oid, side, px, qty, kind, _ = call
            book.submit(oid, side, px, qty, kind, ts)
        elif call[0] == "cancel":
            book.cancel(call[1], ts)
        else:
            _, oid, px, qty, _ = call
            book.modify(oid, px, qty, ts)
        rep.calls += 1
        orders = {s: book.orders(s) for s in Side}
        bid, ask = book.best(Side.Buy), book.best(Side.Sell)
        want = (
            NO_BID if bid is None else bid.price,
            0 if bid is None else bid.qty,
            NO_ASK if ask is None else ask.price,
            0 if ask is None else ask.qty,
        )
        if quote_tuple(spread.quote) != want or quote_tuple(ofi.quote) != want:
            rep.problem(f"call {ts}: quote {spread.quote} / {ofi.quote}, book {want}")

        # Each call's end state lasts one time unit, once the stream has begun.
        if book.event_count:
            if bid is not None and ask is not None:
                spread_bf[ask.price - bid.price] += 1
            else:
                one_sided_bf += 1
            for s in Side:
                if not orders[s]:
                    continue
                observed_bf[s] += 1
                best = orders[s][0].price
                for o in orders[s]:
                    k = abs(o.price - best)
                    if k < SYN_TICKS:
                        depth_bf[s][k] += o.qty

        # Rebuilt L3, and every queue position every 25 calls.
        if ts % 10 == 0:
            for s in Side:
                if queue.orders(s) != [(o.id, o.price, o.qty) for o in orders[s]]:
                    rep.problem(f"call {ts}: {s.name} L3 mirror differs from the book")
        for s in Side:
            ahead_n, ahead_q, prev_px = 0, 0, None
            for o in orders[s]:
                if o.price != prev_px:
                    ahead_n, ahead_q, prev_px = 0, 0, o.price
                pos = (ahead_n, ahead_q, o.qty)
                if ts % 25 == 0:
                    rep.positions += 1
                    if queue.ahead(o.id) != QueuePosition((ahead_n, ahead_q)):
                        rep.problem(f"call {ts}: order {o.id} ahead {queue.ahead(o.id)}")
                if o.id in last:
                    if last[o.id] != pos:
                        last[o.id] = pos
                        history_bf.append((ts, o.id, *pos, int(QueueStatus.Resting)))
                elif o.id % 7 == 0 and o.ts == ts:  # just rested: follow it from here
                    queue.track(o.id)
                    last[o.id] = pos
                    history_bf.append((ts, o.id, *pos, int(QueueStatus.Resting)))
                ahead_n += 1
                ahead_q += o.qty
        live = {o.id for s in Side for o in orders[s]}
        for oid in [i for i in last if i not in live]:
            cancelled = call[0] == "cancel" and call[1] == oid
            status = QueueStatus.Cancelled if cancelled else QueueStatus.Filled
            history_bf.append((ts, oid, 0, 0, 0, int(status)))
            del last[oid]

    now = rep.calls + 1
    if spread.histogram(now) != sorted(spread_bf.items()):
        rep.problem("spread histogram differs from brute force")
    if spread.one_sided_ns(now) != one_sided_bf:
        rep.problem("one-sided time differs from brute force")
    for s in Side:
        if depth.sums(s, now) != depth_bf[s] or depth.observed_ns(s, now) != observed_bf[s]:
            rep.problem(f"{s.name} depth sums differ from brute force")

    rec = ofi.records()
    rep.quote_updates = len(rec["e"])
    quotes = list(zip(rec["bid"], rec["bid_qty"], rec["ask"], rec["ask_qty"], strict=True))
    es = [ofi_e(p, c) for p, c in zip([EMPTY_QUOTE, *quotes[:-1]], quotes, strict=True)]
    if es != rec["e"].tolist():
        rep.problem("an e_n differs from the formula applied to the recorded quotes")
    if ofi.total != sum(es) or ofi.window_sum != sum(es[-SYN_WINDOW:]):
        rep.problem(f"OFI total/window {ofi.total}/{ofi.window_sum} vs {sum(es)}")

    h = queue.history()
    got = sorted(zip(*(h[k].tolist() for k in
                       ("ts", "id", "orders_ahead", "qty_ahead", "leaves", "status")), strict=True))
    rep.samples = len(got)
    if got != sorted(history_bf):
        rep.problem(f"tracked histories differ: {len(got)} vs {len(history_bf)} samples")

    summary = {
        "spread": spread.histogram(now),
        "depth": [depth.sums(s, now) for s in Side],
        "ofi": rec["e"].tolist(),
        "history": got,
    }
    return rep, summary


# ---------------------------------------------------------------------------------------------
# 2. Real day.


@dataclass
class DayResult:
    events: int
    seconds: float
    spread: list
    spread_file: list
    depth_sums: dict
    depth_file: dict
    observed: dict
    observed_file: dict
    ofi_total: int
    ofi_file: int
    quote_updates: int
    file_quote_changes: int
    records: dict
    history: dict
    positions_checked: int
    position_problems: list
    t0: int
    t1: int
    mismatches: int


def snapshot_quote(s) -> tuple:
    bid = s.bids[0] if s.bids else (NO_BID, 0)
    ask = s.asks[0] if s.asks else (NO_ASK, 0)
    return (bid[0], bid[1], ask[0], ask[1])


def replay_with_analytics(path: Path, track_every: int = 40, track_max: int = 1500,
                          check_every: int = 5000) -> DayResult:
    info = lb.describe_file(str(path))
    rp = lb.ArrayReplay(info.levels, CENT)
    spread, depth = SpreadDistribution(CENT), DepthProfile(CENT, DAY_TICKS)
    ofi, queue = OrderFlowImbalance(DAY_WINDOW, record=True), QueuePositions()
    for listener in (spread, depth, ofi, queue):
        rp.attach(listener)

    # The same statistics straight from LOBSTER's orderbook file.
    spread_file: dict[int, int] = defaultdict(int)
    depth_file = {s: [0] * DAY_TICKS for s in Side}
    observed_file = dict.fromkeys(Side, 0)
    ofi_file = 0
    file_changes = 0

    t_start = time.perf_counter()
    reader = iter(lb.DayReader(str(path)))
    m0, s0 = next(reader)
    rp.seed(s0, m0.ns)
    ofi.reset()  # order flow is counted from the seeded book on, as in the file
    state, t_prev, q_prev = s0, m0.ns, snapshot_quote(s0)
    joined = tracked = 0
    checked, problems, n = 0, [], 0
    for m, s in reader:
        n += 1
        dt = m.ns - t_prev
        if dt:
            if state.bids and state.asks:
                spread_file[(state.asks[0][0] - state.bids[0][0]) // CENT] += dt
            for side, levels in ((Side.Buy, state.bids), (Side.Sell, state.asks)):
                if not levels:
                    continue
                observed_file[side] += dt
                best = levels[0][0]
                for px, qty in levels:
                    k = abs(px - best) // CENT
                    if k < DAY_TICKS:
                        depth_file[side][k] += qty * dt
        t_prev = m.ns
        # Sample orders that join the queue at the best price on their side.
        if m.event == lb.Event.Submit and tracked < track_max:
            q = ofi.quote
            best = q.bid if m.side == Side.Buy else q.ask
            if best is not None and m.price == best:
                joined += 1
                if joined % track_every == 0:
                    queue.track(m.id)
                    tracked += 1
        rp.step(m, s)
        q_now = snapshot_quote(s)
        if q_now != q_prev:
            file_changes += 1
            ofi_file += ofi_e(q_prev, q_now)
            q_prev = q_now
        state = s
        if n % check_every == 0:
            book = rp.book
            for side in Side:
                levels = {}
                for o in book.orders(side):
                    levels.setdefault(o.price, []).append(o)
                for queue_orders in levels.values():
                    ahead_n = ahead_q = 0
                    for o in queue_orders:
                        if queue.ahead(o.id) != QueuePosition((ahead_n, ahead_q)):
                            problems.append(f"event {n}: order {o.id}")
                        checked += 1
                        ahead_n += 1
                        ahead_q += o.qty
    seconds = time.perf_counter() - t_start
    now = t_prev
    return DayResult(
        events=n, seconds=seconds,
        spread=spread.histogram(now), spread_file=sorted(spread_file.items()),
        depth_sums={s: depth.sums(s, now) for s in Side}, depth_file=depth_file,
        observed={s: depth.observed_ns(s, now) for s in Side}, observed_file=observed_file,
        ofi_total=ofi.total, ofi_file=ofi_file,
        quote_updates=ofi.updates, file_quote_changes=file_changes,
        records=ofi.records(), history=queue.history(),
        positions_checked=checked, position_problems=problems[:5],
        t0=m0.ns, t1=now, mismatches=rp.stats.mismatches,
    )


def spread_summary(hist: list) -> dict:
    """Mean, median, mode and 10th/90th percentiles of a time-weighted spread histogram."""
    s = np.array([x for x, _ in hist])
    w = np.array([ns for _, ns in hist], dtype=float)
    w /= w.sum()
    c = np.cumsum(w)
    pct = {q: int(s[np.searchsorted(c, q)]) for q in (0.1, 0.5, 0.9)}
    return {"mean": float(s @ w), "median": pct[0.5], "p10": pct[0.1], "p90": pct[0.9],
            "mode": int(s[w.argmax()]), "mode_share": float(w.max()), "max": int(s.max())}


# ---------------------------------------------------------------------------------------------
# OFI regression in fixed time buckets.


@dataclass
class Regression:
    bucket_s: float
    n: int
    beta: float  # cents per share
    se_hc1: float
    r2: float
    r2_lo: float
    r2_hi: float
    ofi: np.ndarray
    dmid: np.ndarray


def ofi_regression(records: dict, t0: int, t1: int, bucket_s: float = 10.0,
                   boot: int = 2000, seed: int = 20261121) -> Regression:
    """Mid-price change (cents) on OFI (shares) over consecutive `bucket_s` intervals."""
    ts = records["ts"].astype(np.int64)
    e = records["e"]
    two = (records["bid"] != NO_BID) & (records["ask"] != NO_ASK)
    mid = np.where(two, (records["bid"] + records["ask"]) / 2.0, np.nan) / CENT
    width = int(bucket_s * 1e9)
    edges = np.arange(t0, t1 + width, width)
    k = np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, len(edges) - 2)
    ofi = np.bincount(k, weights=e, minlength=len(edges) - 1)
    # Mid at each bucket's end: the last quote at or before it (carried forward).
    last_idx = np.searchsorted(ts, edges[1:], side="right") - 1
    mid_end = np.where(last_idx >= 0, mid[np.maximum(last_idx, 0)], np.nan)
    first_idx = np.searchsorted(ts, edges[:-1], side="right") - 1
    mid_start = np.where(first_idx >= 0, mid[np.maximum(first_idx, 0)], np.nan)
    dmid = mid_end - mid_start
    ok = np.isfinite(dmid)
    x, y = ofi[ok], dmid[ok]
    xc, yc = x - x.mean(), y - y.mean()
    beta = float((xc @ yc) / (xc @ xc))
    resid = yc - beta * xc
    n = len(x)
    se = float(np.sqrt(n / (n - 2) * np.sum(xc**2 * resid**2)) / (xc @ xc))
    r2 = float(1 - (resid @ resid) / (yc @ yc))
    rng = np.random.default_rng(seed)
    r2s = np.empty(boot)
    for b in range(boot):
        i = rng.integers(0, n, n)
        xb, yb = x[i] - x[i].mean(), y[i] - y[i].mean()
        bb = (xb @ yb) / (xb @ xb)
        rb = yb - bb * xb
        r2s[b] = 1 - (rb @ rb) / (yb @ yb)
    lo, hi = np.percentile(r2s, [2.5, 97.5])
    return Regression(bucket_s, n, beta, se, r2, float(lo), float(hi), x, y)


def queue_summary(history: dict) -> dict:
    ts, ids, st = history["ts"].astype(np.int64), history["id"], history["status"]
    first: dict[int, int] = {}
    first_ahead: dict[int, int] = {}
    outcome: dict[int, tuple[int, int]] = {}
    for i in range(len(ts)):
        oid = int(ids[i])
        if oid not in first:
            first[oid] = ts[i]
            first_ahead[oid] = int(history["qty_ahead"][i])
        if st[i] != QueueStatus.Resting:
            outcome[oid] = (int(st[i]), ts[i])
    filled = [o for o, (s, _) in outcome.items() if s == QueueStatus.Filled]
    to_fill = np.array([(outcome[o][1] - first[o]) / 1e9 for o in filled])
    return {
        "tracked": len(first),
        "filled": len(filled),
        "cancelled": sum(1 for s, _ in outcome.values() if s == QueueStatus.Cancelled),
        "open": len(first) - len(outcome),
        "median_fill_s": float(np.median(to_fill)) if len(to_fill) else float("nan"),
        "median_ahead": float(np.median(list(first_ahead.values()))) if first_ahead else 0.0,
        "first": first,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------------------------
# 3. Figure.


def plot(day: DayResult, reg: Regression, qs: dict, title_day: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    # Spread distribution.
    ax = axes[0, 0]
    sp = spread_summary(day.spread)
    total = sum(ns for _, ns in day.spread)
    cap = 35
    xs = [x for x, _ in day.spread if x <= cap]
    ys = [100 * ns / total for x, ns in day.spread if x <= cap]
    beyond = 100 * sum(ns for x, ns in day.spread if x > cap) / total
    ax.bar(xs, ys, color=blue, width=0.8)
    ax.axvline(sp["median"], color=orange, lw=1.5, ls="--")
    ax.text(sp["median"] + 0.6, max(ys) * 0.98, f"median {sp['median']}¢", color=orange,
            fontsize=8, va="top")
    ax.text(0.98, 0.70, f"wider than {cap}¢: {beyond:.1f}% of the time\nwidest {sp['max']}¢",
            transform=ax.transAxes, ha="right", color=ink2, fontsize=7.5)
    style(ax, f"Spread: median {sp['median']}¢, between {sp['p10']}¢ and {sp['p90']}¢ "
          "80% of the time", "Bid-ask spread (cents)", "Share of two-sided time (%)")

    # Depth profile.
    ax = axes[0, 1]
    k = np.arange(DAY_TICKS)
    ratios = []
    for side, color, label in ((Side.Buy, blue, "Bids"), (Side.Sell, orange, "Asks")):
        mean_depth = np.array(day.depth_sums[side]) / day.observed[side]
        ratios.append(mean_depth[0] / mean_depth[1:].mean())
        ax.plot(k, mean_depth, color=color, lw=2, marker="o", ms=4, mec=surface, label=label)
    style(ax, f"Depth: the best price holds ~{np.mean(ratios):.1f}× each price behind it",
          "Ticks (cents) from the best price on that side",
          "Mean resting size (shares; empty prices count as 0)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink)

    # OFI vs mid change.
    ax = axes[1, 0]
    ax.scatter(reg.ofi, reg.dmid, s=6, color=blue, alpha=0.35, lw=0)
    xx = np.linspace(np.percentile(reg.ofi, 0.5), np.percentile(reg.ofi, 99.5), 50)
    ax.plot(xx, reg.beta * (xx - reg.ofi.mean()) + reg.dmid.mean(), color=orange, lw=2)
    ax.set_xlim(xx[0], xx[-1])
    style(ax, f"Order-flow imbalance explains {100 * reg.r2:.0f}% of 10 s price moves",
          "OFI over the interval (shares)", "Mid-price change (cents)")
    ax.text(0.02, 0.96, f"slope {reg.beta * 1e3:.2f} ± {reg.se_hc1 * 1e3:.2f}¢ per 1,000 shares "
            f"(HC1 s.e.)\nR² {reg.r2:.2f}, 95% bootstrap {reg.r2_lo:.2f}–{reg.r2_hi:.2f}, "
            f"{reg.n:,} intervals", transform=ax.transAxes, va="top", color=ink2, fontsize=7.5)

    # Queue positions, as a share of the queue ahead on joining.
    ax = axes[1, 1]
    h = day.history
    ts, ids, qa = h["ts"].astype(np.int64), h["id"], h["qty_ahead"]
    filled = sorted(o for o, (st, _) in qs["outcome"].items() if st == QueueStatus.Filled)
    rng = np.random.default_rng(7)
    shown_ids = rng.choice(filled, size=min(40, len(filled)), replace=False).tolist()
    for oid in shown_ids:
        sel = ids == oid
        a = qa[sel].astype(float)
        if a[0] <= 0:
            continue
        a[h["status"][sel] != QueueStatus.Resting] = 0
        t = (ts[sel] - qs["first"][oid]) / 1e9
        ax.step(t, a / a[0], where="post", color=green, lw=1, alpha=0.55)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.03, 1.05)
    style(ax, "The queue ahead drains to zero before a fill",
          "Seconds since joining at the best price (symlog)",
          "Shares ahead / shares ahead on joining")
    ax.text(0.98, 0.18, f"{len(shown_ids)} of {qs['filled']:,} filled orders shown\n"
            f"{qs['tracked']:,} sampled: {qs['filled']:,} filled, {qs['cancelled']:,} cancelled"
            f"\nmedian {qs['median_ahead']:,.0f} shares ahead on joining, "
            f"{qs['median_fill_s']:.1f} s to fill",
            transform=ax.transAxes, ha="right", va="bottom", color=ink2, fontsize=7.5,
            bbox={"facecolor": surface, "edgecolor": grid, "boxstyle": "round,pad=0.4"})

    fig.suptitle(f"{title_day}: microstructure from the engine's event stream",
                 x=0.01, ha="left", color=ink, fontsize=12, fontweight="bold")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calls", type=int, default=4000, help="calls per synthetic regime")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--lobster", type=Path, help="LOBSTER message file (default: first in data/)")
    args = ap.parse_args(argv)
    failures: list[str] = []

    print(f"## Synthetic flows: analytics vs brute force, {args.calls:,} calls per regime\n")
    print("| Regime | Quote updates | Positions checked | Tracked samples | Problems (map/array) |")
    print("|---|---:|---:|---:|---|")
    for j, regime in enumerate(REGIMES):
        reps, sums = {}, {}
        for name, cls in BOOKS.items():
            reps[name], sums[name] = run_synthetic(regime, args.seed + j, args.calls, cls)
            for p in reps[name].problems:
                failures.append(f"{regime.name} ({name}): {p}")
        if sums["map"] != sums["array"]:
            failures.append(f"{regime.name}: map and array analytics differ")
        r = reps["map"]
        print(f"| {r.regime} | {r.quote_updates:,} | {r.positions:,} | {r.samples:,} | "
              f"{len(reps['map'].problems)} / {len(reps['array'].problems)} |")
    print("\nChecked after every call: rebuilt quote; spread and depth charged per unit of time; "
          "rebuilt L3 every 10 calls; every queue position every 25 calls; tracked histories; "
          "each e_n from the recorded quotes; window and total. Map and array: identical.")

    day = args.lobster or find_lobster_day()
    if day is None:
        print("\n## Real LOBSTER day\n\nNo LOBSTER file in data/ (expected in CI); skipped.")
    else:
        res = replay_with_analytics(day)
        print(f"\n## Real day: {day.name}, analytics vs LOBSTER's own orderbook file\n")
        print("| Statistic | From the engine's events | From LOBSTER's file | Match |")
        print("|---|---|---|---|")
        two_sided = sum(ns for _, ns in res.spread)
        sp = spread_summary(res.spread)
        top = (f"mean {sp['mean']:.1f}¢, median {sp['median']}¢ (10–90%: {sp['p10']}–"
               f"{sp['p90']}¢), most often {sp['mode']}¢ ({100 * sp['mode_share']:.1f}%)")
        same = res.spread == res.spread_file
        print(f"| Spread, time-weighted ({two_sided / 1e9:,.0f} s two-sided) | {top} | "
              f"{'same' if same else 'differs'} | {'exact' if same else '**NO**'} |")
        if not same:
            failures.append("spread distribution differs from LOBSTER's file")
        for side in Side:
            same = (res.depth_sums[side] == res.depth_file[side]
                    and res.observed[side] == res.observed_file[side])
            mean = np.array(res.depth_sums[side]) / res.observed[side]
            print(f"| Depth at 0–9 ticks, {side.name.lower()}s (mean shares) | "
                  f"{', '.join(f'{v:.0f}' for v in mean)} | {'same' if same else 'differs'} | "
                  f"{'exact' if same else '**NO**'} |")
            if not same:
                failures.append(f"{side.name} depth profile differs from LOBSTER's file")
        same = res.ofi_total == res.ofi_file
        print(f"| Cumulative OFI (shares) | {res.ofi_total:+,} over {res.quote_updates:,} quote "
              f"updates | {res.ofi_file:+,} over {res.file_quote_changes:,} quote changes | "
              f"{'exact' if same else '**NO**'} |")
        if not same:
            failures.append("cumulative OFI differs from LOBSTER's file")
        ok = not res.position_problems
        print(f"| Queue positions vs the replayed book's L3 view | {res.positions_checked:,} "
              f"orders at {res.events // 5000} checkpoints | – | "
              f"{'exact' if ok else '**NO**'} |")
        if not ok:
            failures.append(f"queue positions differ: {res.position_problems}")
        if res.mismatches:
            failures.append(f"replay mismatches: {res.mismatches}")
        print(f"\nOne Python loop replays the day with the four analytics attached and recomputes "
              f"the same statistics from LOBSTER's file: {res.events:,} events in "
              f"{res.seconds:.1f} s.")

        reg = ofi_regression(res.records, res.t0, res.t1)
        qs = queue_summary(res.history)
        l1 = (np.array(res.depth_sums[Side.Buy][0]) / res.observed[Side.Buy]
              + np.array(res.depth_sums[Side.Sell][0]) / res.observed[Side.Sell]) / 2
        print("\n## OFI and price changes, 10-second intervals\n")
        print("| Intervals | Slope (¢ per 1,000 shares) | HC1 s.e. | R² | R² 95% bootstrap | "
              "1/(2·slope) | Mean depth at the best |")
        print("|---:|---:|---:|---:|---|---:|---:|")
        print(f"| {reg.n:,} | {reg.beta * 1e3:.3f} | {reg.se_hc1 * 1e3:.3f} | {reg.r2:.3f} | "
              f"{reg.r2_lo:.3f}–{reg.r2_hi:.3f} | {1 / (2 * reg.beta):,.0f} shares | "
              f"{l1:,.0f} shares |")
        print("\n## Queue positions of sampled orders that joined at the best\n")
        print(f"{qs['tracked']:,} tracked: {qs['filled']:,} filled, {qs['cancelled']:,} "
              f"cancelled, {qs['open']:,} still resting at the close. Median shares ahead on "
              f"joining {qs['median_ahead']:,.0f}; median time to fill "
              f"{qs['median_fill_s']:.1f} s.")
        if not args.no_figure:
            name = day.name.split("_")
            plot(res, reg, qs, f"{name[0]} {name[1]}")
            print(f"\nFigure: {FIGURE.relative_to(ROOT)}")

    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for f in failures:
        print(f"  - {f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
