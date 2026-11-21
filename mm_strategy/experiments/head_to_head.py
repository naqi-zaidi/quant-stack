"""Project 3 step 7: the head-to-head, and the Phase 3 gate.

The roadmap's gate for the market maker is walk-forward, out-of-sample results net of fees.
This runs every strategy of steps 2-6 in every environment, from one command.

Strategies (100 shares a side; maker rebate $0.002 a share in every PnL):
  fixed 1 tick, fixed 3 ticks, join the best (quote at the best bid and ask), A–S (γ = 3e-4),
  A–S + flow (step 4's signal), and A–S + flow with step 6's risk overlay.
Environments:
  synthetic, live      seeds 1-N, each strategy trading its own session (steps 2-4)
  synthetic, replayed  the same seeds' sessions without a maker, recorded as tapes and replayed
                       with queue-aware virtual orders (step 5)
  synthetic, stressed  the same seeds with one hidden 5-minute stress episode, live (step 6)
  AAPL, replayed       2012-06-21, 11 half-hour blocks from 10:30 (step 5; --aapl, local only)

Nothing is fitted on the session it is scored on. On the synthetic market, σ, k, the k-means
regimes and the kill threshold come from the history block (seeds 1001-1010), and each
environment keeps its own walk-forward chain for the flow model: day d's model is fitted on the
history block and that environment's days 1..d-1. On AAPL, σ, k and the flow model are refitted
before every block on the blocks before it; the regimes and kill threshold come from 9:30-10:30.

The gate checks what must hold exactly: the PnL identity everywhere; every day's flow model
refits exactly from the logs before it, in every chain; with no maker, the replay shows a
strategy exactly the live market; the overlay's cap and kill switch; and, with --aapl, a
replayed book that matches LOBSTER after every message. PnL is reported with standard errors.
The README's tables come from --sessions 40 --aapl; CI runs --sessions 2.

    uv run python -m mm_strategy.experiments.head_to_head [--sessions 2] [--aapl] [--no-figure]
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np

from mm_strategy.accounting import decompose, inventory_stats
from mm_strategy.calibration import estimate_intensity, estimate_sigma
from mm_strategy.experiments.avellaneda import CAL_SEEDS, mean_se, paired
from mm_strategy.experiments.blind_spot import Recorder
from mm_strategy.experiments.flow import GAMMA, HORIZON_S, WEIGHT, history_block
from mm_strategy.experiments.risk import (
    BAR_S,
    CAP,
    STRESS,
    STRESS_LEN_S,
    regimes_from,
    stress_start,
)
from mm_strategy.experiments.risk_aapl import drawdown as replay_drawdown
from mm_strategy.maker import AvellanedaStoikov, FixedSpread, JoinBest, MarketMaker
from mm_strategy.market import MarketParams, simulate
from mm_strategy.replay import ReplayMaker, TapeReplay, TapeSource
from mm_strategy.risk import RiskOverlay
from mm_strategy.signals import FlowSkewAS, fit

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_head_to_head.png"
DOLLAR = 10_000
NAMES = ("fixed 1 tick", "fixed 3 ticks", "join the best", "A–S", "A–S + flow",
         "A–S + flow + overlay")
FLOW, GUARD = "A–S + flow", "A–S + flow + overlay"
ENVS = ("synthetic, live", "synthetic, replayed", "synthetic, stressed", "AAPL, replayed")


def strategies(base, model, regimes, kill, warm=None) -> dict:
    guard = RiskOverlay(FlowSkewAS(base, model, WEIGHT), regimes, bar_s=BAR_S, inventory_cap=CAP,
                        max_drawdown=kill)
    if warm is not None:
        guard.warm_up(warm)
    return {"fixed 1 tick": FixedSpread(1), "fixed 3 ticks": FixedSpread(3),
            "join the best": JoinBest(), "A–S": base, FLOW: FlowSkewAS(base, model, WEIGHT),
            GUARD: guard}


def checks_of(strategy, inv: np.ndarray, samples) -> dict:
    out = {}
    if isinstance(strategy, RiskOverlay):
        out["cap_ok"] = int(np.abs(inv).max()) <= CAP + 100
        if strategy.killed_at is not None:
            s = np.array(samples, dtype=float)
            out["kill_ok"] = bool(np.all(np.diff(np.abs(s[s[:, 0] >= strategy.killed_at, 1])) <= 0))
    return out


def live(p, seed, strategy) -> dict:
    mm = MarketMaker(strategy)
    s = simulate(p, seed, participant=mm)
    pnl = decompose(s, mm)
    inv = np.array([x[1] for x in mm.samples])
    return {"net": pnl.net / DOLLAR, "volume": pnl.volume, "identity": pnl.identity_holds,
            "drawdown": inventory_stats(s, mm)["max_drawdown"] / DOLLAR,
            "abs_q": float(np.abs(inv).mean()), **checks_of(strategy, inv, mm.samples)}


def replayed(rm: ReplayMaker, quotes, scale: float = 1.0) -> dict:
    pnl = rm.pnl()
    inv = np.array([x[1] for x in rm.samples])
    return {"net": scale * pnl.net / DOLLAR, "volume": scale * pnl.volume,
            "identity": pnl.identity_holds, "drawdown": replay_drawdown(rm, quotes),
            "abs_q": float(np.abs(inv).mean()), **checks_of(rm.strategy, inv, rm.samples)}


def flow_log(strategy) -> list:
    return strategy.features.log


def synthetic(n: int) -> tuple[dict, dict]:
    p = MarketParams()
    cal = [simulate(p, sd) for sd in CAL_SEEDS]
    base = AvellanedaStoikov(GAMMA, estimate_sigma(cal, 60.0).value,
                             estimate_intensity(cal).k.value)
    hist_logs, hist = history_block(p, base)
    regimes = regimes_from(hist_logs)
    kill = 2 * max(r["drawdown"] for r in hist) * DOLLAR
    res = {e: {k: [] for k in NAMES} for e in ENVS[:3]}
    chains = {e: [] for e in ENVS[:3]}
    models = {e: [] for e in ENVS[:3]}
    faithful = 0
    for sd in range(1, n + 1):
        m = {e: fit(hist_logs + chains[e], HORIZON_S) for e in ENVS[:3]}
        for e in ENVS[:3]:
            models[e].append(m[e])
        # Live, and the tape of the same session without a maker.
        rec_live = Recorder()
        tape = simulate(p, sd, participant=MarketMaker(rec_live), record_tape=True).tape
        st = strategies(base, m["synthetic, live"], regimes, kill)
        for k, s in st.items():
            res["synthetic, live"][k].append(live(p, sd, s))
        chains["synthetic, live"].append(flow_log(st[FLOW]))
        # Replayed.
        st = strategies(base, m["synthetic, replayed"], regimes, kill)
        makers = {k: ReplayMaker(s, 0, p.duration_ns) for k, s in st.items()}
        rec = Recorder()
        rp = TapeReplay(TapeSource(tape))
        rp.run([*makers.values(), ReplayMaker(rec, 0, p.duration_ns)], p.duration_ns)
        faithful += rec.views == rec_live.views
        q = rp.quotes()
        for k, rm in makers.items():
            res["synthetic, replayed"][k].append(replayed(rm, q))
        chains["synthetic, replayed"].append(flow_log(st[FLOW]))
        # Stressed.
        ps = dataclasses.replace(p, stress_start_s=stress_start(sd), **STRESS)
        st = strategies(base, m["synthetic, stressed"], regimes, kill)
        for k, s in st.items():
            res["synthetic, stressed"][k].append(live(ps, sd, s))
        chains["synthetic, stressed"].append(flow_log(st[FLOW]))
    clean = all(fit(hist_logs + chains[e][:i], HORIZON_S) == mm
                and mm.sessions == len(CAL_SEEDS) + i
                for e in ENVS[:3] for i, mm in enumerate(models[e]))
    info = {"faithful": faithful, "n": n, "clean": clean, "kill": kill / DOLLAR,
            "stress_len": STRESS_LEN_S}
    return res, info


def aapl() -> tuple[dict, dict]:
    from mm_strategy.experiments.aapl import (
        DATA,
        DELTAS,
        HISTORY,
        N_BLOCKS,
        Day,
        FeatureLogger,
        block_bounds,
    )
    from mm_strategy.replay import LobsterSource

    if not DATA.exists():
        raise SystemExit(f"{DATA.relative_to(ROOT)} not found: see data/README.md")
    day = Day(DATA)
    windows = [day.block(i) for i in range(N_BLOCKS)]
    rp = TapeReplay(LobsterSource(str(DATA)))
    sig_h = estimate_sigma(windows[:HISTORY], 60.0).value
    k_h = estimate_intensity(windows[:HISTORY], DELTAS).k.value
    logs, hist_dd = [], []
    for i in range(HISTORY):
        start, end = block_bounds(i)
        logger = FeatureLogger()
        plain = ReplayMaker(AvellanedaStoikov(GAMMA, sig_h, k_h), start, end)
        rp.run([ReplayMaker(logger, start, end), plain], end)
        logs.append(logger.features.log)
        hist_dd.append(replay_drawdown(plain, rp.quotes()))
    regimes = regimes_from(logs)
    kill = 2 * max(hist_dd) * DOLLAR
    res = {k: [] for k in NAMES}
    models = []
    for i in range(HISTORY, N_BLOCKS):
        start, end = block_bounds(i)
        base = AvellanedaStoikov(GAMMA, estimate_sigma(windows[:i], 60.0).value,
                                 estimate_intensity(windows[:i], DELTAS).k.value)
        model = fit(logs, HORIZON_S)
        models.append((model, len(logs)))
        st = strategies(base, model, regimes, kill, warm=np.asarray(logs[-1], float)[:, 1])
        makers = {k: ReplayMaker(s, start, end) for k, s in st.items()}
        logger = FeatureLogger()
        rp.run([*makers.values(), ReplayMaker(logger, start, end)], end)
        logs.append(logger.features.log)
        q = rp.quotes()
        for k, rm in makers.items():
            res[k].append(replayed(rm, q, scale=2.0))  # per hour: blocks are 30 minutes
    clean = all(fit(logs[:n], HORIZON_S) == m for m, n in models)
    st_ = rp.rp.stats
    info = {"replay_ok": st_.mismatches == 0 and st_.errors == 0, "events": st_.events,
            "clean": clean, "kill": kill / DOLLAR, "n": N_BLOCKS - HISTORY}
    return res, info


def col(rows, key) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def plot(res: dict, envs: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
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

    # Net PnL, every strategy in every environment.
    ax = axes[0, 0]
    y = np.arange(len(NAMES))[::-1]
    for j, e in enumerate(envs):
        off = (j - (len(envs) - 1) / 2) * 0.18
        m = np.array([mean_se(col(res[e][k], "net")) for k in NAMES])
        ax.errorbar(m[:, 0], y + off, xerr=1.96 * m[:, 1], fmt="o", ms=4, color=palette[j],
                    capsize=2, lw=1.1, label=e)
    ax.axvline(0, color=ink2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(NAMES, fontsize=8)
    wins = {(k, e) for k in NAMES for e in envs
            if (lambda m: m[0] - 4 * m[1] > 0)(mean_se(col(res[e][k], "net")))}
    if wins and all(k.startswith(FLOW) for k, _ in wins) and \
            all(e in ("synthetic, live", "synthetic, replayed") for _, e in wins):
        title = "4σ profit: A–S + flow, calm synthetic market only"
    else:
        title = f"Profitable at 4σ in {len(wins)} of {len(NAMES) * len(envs)} cases"
    style(ax, title, "Net PnL ($/hour), 95% CI", "")
    ax.legend(frameon=False, fontsize=7, labelcolor=ink, loc="lower left")

    # Risk and return, synthetic live.
    ax = axes[0, 1]
    e = "synthetic, live"
    for k in NAMES:
        net = col(res[e][k], "net")
        sd, (m, se) = float(np.std(net, ddof=1)), mean_se(net)
        ax.errorbar(sd, m, yerr=1.96 * se, fmt="o", ms=5, color=palette[0], capsize=2, lw=1)
        dx, dy, ha = {GUARD: (0, 9, "center"), FLOW: (8, -12, "left")}.get(k, (8, -3, "left"))
        ax.annotate(k, (sd, m), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    color=ink2, ha=ha)
    ax.axhline(0, color=ink2, lw=0.8)
    style(ax, "Synthetic, live: skew cuts the risk, flow adds the return",
          "PnL standard deviation across sessions ($/hour)", "Mean net PnL ($/hour), 95% CI")

    # The same strategies: live → replayed → AAPL.
    ax = axes[1, 0]
    chain = [x for x in ("synthetic, live", "synthetic, replayed", "AAPL, replayed") if x in envs]
    for k, color in (("fixed 3 ticks", palette[1]), ("A–S", palette[2]), (GUARD, palette[0])):
        m = np.array([mean_se(col(res[x][k], "net")) for x in chain])
        ax.errorbar(range(len(chain)), m[:, 0], yerr=1.96 * m[:, 1], color=color, marker="o",
                    ms=4, capsize=2, lw=1.5, label=k)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(range(len(chain)))
    ax.set_xticklabels(chain, fontsize=8)
    style(ax, "What survives the move to real data", "", "Net PnL ($/hour), 95% CI")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="lower left")

    # Paired gains: the flow signal and the overlay, per environment.
    ax = axes[1, 1]
    x = np.arange(len(envs))
    for j, (a, b, label) in enumerate(((FLOW, "A–S", "flow signal (A–S + flow − A–S)"),
                                       (GUARD, FLOW, "overlay (+ overlay − A–S + flow)"))):
        d = np.array([paired(col(res[e][a], "net"), col(res[e][b], "net")) for e in envs])
        ax.bar(x + (j - 0.5) * 0.36, d[:, 0], width=0.34, color=palette[j], label=label)
        ax.errorbar(x + (j - 0.5) * 0.36, d[:, 0], yerr=1.96 * d[:, 1], fmt="none", color=ink,
                    capsize=2, lw=1)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(", ", ",\n") for e in envs], fontsize=7.5)
    style(ax, "The flow signal pays on the synthetic market, not on AAPL", "",
          "Paired gain ($/hour), 95% CI")
    ax.legend(frameon=False, fontsize=7, labelcolor=ink, loc="upper right")

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--aapl", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    failures: list[str] = []
    res, info = synthetic(args.sessions)
    envs = list(ENVS[:3])
    if args.aapl:
        res["AAPL, replayed"], ainfo = aapl()
        envs.append("AAPL, replayed")

    rows = [r for e in envs for k in NAMES for r in res[e][k]]
    checks = {
        "PnL identity, every strategy in every environment": all(r["identity"] for r in rows),
        "every day's flow model refits exactly from the logs before it, in every chain":
            info["clean"] and (not args.aapl or ainfo["clean"]),
        f"with no maker, the replay shows exactly the live market ({info['faithful']} of "
        f"{info['n']} sessions)": info["faithful"] == info["n"],
        f"overlay: never more than {CAP} + 100 shares": all(r.get("cap_ok", True) for r in rows),
        "overlay: inventory only shrinks after the kill switch":
            all(r.get("kill_ok", True) for r in rows),
    }
    if args.aapl:
        checks[f"AAPL: replayed book matches LOBSTER after all {ainfo['events']:,} messages"] = \
            ainfo["replay_ok"]
    n = info["n"]
    print(f"## Head-to-head: {n} synthetic seeds"
          + (f" and {ainfo['n']} AAPL blocks" if args.aapl else "") + ", walk-forward\n")
    print("| Check | Gate |")
    print("|---|---|")
    for name, ok in checks.items():
        print(f"| {name} | {'yes' if ok else '**NO**'} |")
        if not ok:
            failures.append(name)

    head = " | ".join(envs)
    print(f"\n### Net PnL ($/hour, mean ± s.e.; rebates included)\n\n| Strategy | {head} |")
    print("|---|" + "---:|" * len(envs))
    for k in NAMES:
        cells = []
        for e in envs:
            m, se = mean_se(col(res[e][k], "net"))
            cell = f"{m:+,.0f} ± {se:,.0f}"
            cells.append(f"**{cell}**" if m - 4 * se > 0 else cell)
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("\nBold: above zero at 4 standard errors.")

    print(f"\n### Risk: PnL s.d. across sessions / mean max drawdown ($)\n\n| Strategy | {head} |")
    print("|---|" + "---:|" * len(envs))
    for k in NAMES:
        cells = [f"{np.std(col(res[e][k], 'net'), ddof=1):,.0f} / "
                 f"{col(res[e][k], 'drawdown').mean():,.0f}" for e in envs]
        print(f"| {k} | " + " | ".join(cells) + " |")

    print(f"\n### Paired differences ($/hour)\n\n| Comparison | {head} |")
    print("|---|" + "---:|" * len(envs))
    for a, b in ((FLOW, "A–S"), (GUARD, FLOW), ("A–S", "fixed 3 ticks"), (GUARD, "join the best")):
        cells = []
        for e in envs:
            d, se = paired(col(res[e][a], "net"), col(res[e][b], "net"))
            cells.append(f"{d:+,.0f} ± {se:,.0f} (t {d / se:+.1f})")
        print(f"| {a} − {b} | " + " | ".join(cells) + " |")

    print(f"\nSynthetic: kill switch at ${info['kill']:,.0f}"
          + (f"; AAPL: ${ainfo['kill']:,.0f}" if args.aapl else "") + f"; cap {CAP} shares.")
    if not args.no_figure:
        plot(res, envs)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nRan in {time.time() - t0:.0f} s.")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
