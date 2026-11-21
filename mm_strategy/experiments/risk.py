"""Project 3 step 6 gate: the FYP's risk controller as an overlay on the market maker.

mm_strategy.risk ports the regime detector, anomaly scoring and policy state machine of my
final-year project and adds an inventory cap and a drawdown kill switch. This experiment runs
it on the synthetic market in two arms, paired by seed:
- calm: the market as in steps 1-5 (constant volatility), where every flag is a false alarm;
- stressed: the same seeds with one 5-minute stress episode at a random time between minutes
  10 and 50 (fair-value volatility x3, informed traders x3 as often). The strategy is not told.

Everything the overlay needs comes from the history block (seeds 1001-1010, calm, plain A–S),
as in step 4: σ and k, the flow model, the k-means regimes (on 5-second bars), and the kill
switch's threshold, set at twice the largest drawdown plain A–S had there. The inventory cap is
500 shares. The strategies are A–S + flow (with the history block's flow model, as in the step 5
blind-spot test) and the 1-tick fixed spread, which has no inventory control of its own.
A–S + flow also runs with each part of the overlay alone.

The gate checks what must hold exactly: the PnL identity in every session; no capped maker
ever holds more than the cap plus one quote; after the kill switch fires, inventory only
shrinks. Detection and PnL are reported with paired standard errors, as is the FYP's own
restraint test (calm bars: pause rate under 1%, throttle rate under 5%). The README's tables
come from --sessions 40; CI runs --sessions 2.

    uv run python -m mm_strategy.experiments.risk [--sessions 2] [--no-figure] [--aapl]

--aapl (local only, needs the LOBSTER sample) adds the overlay to the AAPL walk-forward of step
5, with the regimes and the kill threshold fitted on the 9:30-10:30 history block.
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
from mm_strategy.experiments.flow import GAMMA, HORIZON_S, WEIGHT, history_block
from mm_strategy.maker import AvellanedaStoikov, FixedSpread, MarketMaker
from mm_strategy.market import NS, MarketParams, simulate
from mm_strategy.risk import RegimeModel, RiskOverlay, bar_features
from mm_strategy.signals import FlowSkewAS, fit

ROOT = Path(__file__).resolve().parents[2]
FIGURE = ROOT / "docs" / "figures" / "p3_risk.png"
DOLLAR = 10_000
BAR_S = 5.0
PER_BAR = round(BAR_S / 0.1)
WARM_S = 60 * BAR_S  # the features' longest window, 60 bars
CAP = 500
STRESS_LEN_S = 300.0
STRESS = {"stress_len_s": STRESS_LEN_S, "stress_vol": 3.0, "stress_informed": 3.0}
FLOW, FIXED = "A–S + flow", "fixed 1 tick"
VARIANTS = ("none", "policy only", "cap only", "full overlay")


def stress_start(seed: int) -> float:
    """Where the episode falls in a session: minute 10 to 50, from a stream of its own."""
    return float(np.random.default_rng([seed, 6]).uniform(600, 3000 - STRESS_LEN_S))


def regimes_from(logs) -> RegimeModel:
    x = np.vstack([[f.regime_vector()
                    for f in bar_features(np.asarray(lg, float)[:, 1], PER_BAR) if f.ready]
                   for lg in logs])
    return RegimeModel.fit(x)


def overlay(inner, variant: str, regimes, kill):
    if variant == "none":
        return inner
    return RiskOverlay(inner, regimes if variant in ("policy only", "full overlay") else None,
                       bar_s=BAR_S,
                       inventory_cap=CAP if variant in ("cap only", "full overlay") else None,
                       max_drawdown=kill if variant == "full overlay" else None)


def run(p: MarketParams, seed: int, strategy, window: tuple[float, float] | None) -> dict:
    mm = MarketMaker(strategy)
    s = simulate(p, seed, participant=mm)
    pnl = decompose(s, mm)
    inv = inventory_stats(s, mm)
    smp = np.array(mm.samples, dtype=float)
    marked = smp[:, 2] + smp[:, 1] * s.mid_at(smp[:, 0].astype(np.int64))
    out = {"net": pnl.net / DOLLAR, "volume": pnl.volume,
           "drawdown": inv["max_drawdown"] / DOLLAR, "max_q": inv["max_abs_inventory"],
           "identity": pnl.identity_holds}
    if window is not None:
        t = smp[:, 0] / NS
        i0, i1 = np.searchsorted(t, window[0]), np.searchsorted(t, window[1])
        out["stress_pnl"] = float(marked[min(i1, len(t) - 1)] - marked[max(i0 - 1, 0)]) / DOLLAR
    if isinstance(strategy, RiskOverlay):
        o = strategy
        out["cap_ok"] = o.inventory_cap is None or inv["max_abs_inventory"] <= CAP + 100
        out["killed"] = o.killed_at is not None
        if o.killed_at is not None:
            after = np.abs(smp[smp[:, 0] >= o.killed_at, 1])
            out["kill_ok"] = bool(np.all(np.diff(after) <= 0))
        if o.regimes is not None:
            t = np.array([b.ts for b in o.bars]) / NS
            act = np.array([b.action for b in o.bars])
            live = t > WARM_S
            inw = ((t > window[0]) & (t <= window[1]) if window is not None
                   else np.zeros(len(t), bool))
            calm = live & ~inw
            out["throttle_calm"] = float((act[calm] == "throttle").mean())
            out["pause_calm"] = float((act[calm] == "pause").mean())
            out["flag_calm"] = float((act[calm] != "allow").mean())
            if window is not None:
                out["flag_stress"] = float((act[inw] != "allow").mean())
                hit = np.flatnonzero(inw & (act != "allow"))
                out["latency"] = float(t[hit[0]] - window[0]) if len(hit) else np.nan
    return out


def col(rows, key) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def plot(res, n) -> None:
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

    colors = dict(zip(VARIANTS, (ink2, blue, green, orange), strict=True))

    # Detection: flagged share of bars, calm vs stress.
    ax = axes[0, 0]
    pol = res["stressed"][FLOW]["policy only"]
    fc, fs = col(pol, "flag_calm"), col(pol, "flag_stress")
    calm_arm = col(res["calm"][FLOW]["policy only"], "flag_calm")
    groups = (("calm sessions", calm_arm), ("stressed sessions,\noutside the episode", fc),
              ("stressed sessions,\nduring the episode", fs))
    for i, (_label, v) in enumerate(groups):
        m, se = mean_se(v)
        ax.bar(i, m * 100, color=(green, green, orange)[i], width=0.6)
        ax.errorbar(i, m * 100, yerr=1.96 * se * 100, color=ink, capsize=3, lw=1)
    ax.axhline(5, color=ink, lw=1, ls="--", label="the FYP's restraint ceiling: 5% of calm bars")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper left")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
    d, se = paired(fs, fc)
    style(ax, f"The FYP policy flags {fs.mean():.0%} of stress bars and {fc.mean():.0%} of calm "
          "ones", "", "Bars throttled or paused (%), 95% CI")

    # Net PnL by variant, both arms (A–S + flow).
    ax = axes[0, 1]
    for j, arm in enumerate(("calm", "stressed")):
        for i, v in enumerate(VARIANTS):
            m, se = mean_se(col(res[arm][FLOW][v], "net"))
            x = i + (j - 0.5) * 0.36
            ax.bar(x, m, width=0.34, color=colors[v], alpha=1.0 if arm == "stressed" else 0.45)
            ax.errorbar(x, m, yerr=1.96 * se, color=ink, capsize=2, lw=1)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, fontsize=7.5)
    ax.axhline(0, color=ink2, lw=0.8)
    d, se = paired(col(res["stressed"][FLOW]["full overlay"], "net"),
                   col(res["stressed"][FLOW]["none"], "net"))
    style(ax, f"Under stress the overlay adds {d:+,.0f} ± {se:,.0f} \\$/hour to A–S + flow",
          "pale: calm sessions; solid: stressed sessions", "Net PnL ($/hour), 95% CI")

    # Drawdown, fixed spread vs A–S + flow, with and without the overlay (stressed arm).
    ax = axes[1, 0]
    labels, vals, errs, cols = [], [], [], []
    for strat in (FIXED, FLOW):
        for v in ("none", "full overlay"):
            m, se = mean_se(col(res["stressed"][strat][v], "drawdown"))
            labels.append(f"{strat}\n{'no overlay' if v == 'none' else 'overlay'}")
            vals.append(m)
            errs.append(se)
            cols.append(colors[v])
    ax.bar(range(4), vals, color=cols, width=0.6)
    ax.errorbar(range(4), vals, yerr=1.96 * np.array(errs), fmt="none", color=ink, capsize=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7.5)
    r = vals[1] / vals[0]
    style(ax, f"Under stress the overlay cuts the fixed spread's drawdown to {r:.0%}",
          "stressed sessions", "Max drawdown in a session ($), 95% CI")

    # Stress-window PnL by variant (A–S + flow), paired with none.
    ax = axes[1, 1]
    base = col(res["stressed"][FLOW]["none"], "stress_pnl")
    for i, v in enumerate(VARIANTS[1:]):
        d, se = paired(col(res["stressed"][FLOW][v], "stress_pnl"), base)
        ax.bar(i, d, color=colors[v], width=0.6)
        ax.errorbar(i, d, yerr=1.96 * se, color=ink, capsize=3, lw=1)
    ax.axhline(0, color=ink2, lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(VARIANTS[1:], fontsize=7.5)
    style(ax, "What each part saves during the 5-minute episode", "A–S + flow, stressed sessions",
          "PnL during the episode vs no overlay ($), 95% CI")
    _ = yellow

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--aapl", action="store_true")
    args = ap.parse_args(argv)
    if args.aapl:
        from mm_strategy.experiments.risk_aapl import main as aapl_main

        aapl_main(args.no_figure)
        return
    p = MarketParams()
    failures: list[str] = []
    t0 = time.time()

    cal = [simulate(p, sd) for sd in CAL_SEEDS]
    base = AvellanedaStoikov(GAMMA, estimate_sigma(cal, 60.0).value,
                             estimate_intensity(cal).k.value)
    logs, hist = history_block(p, base)
    model = fit(logs, HORIZON_S)
    regimes = regimes_from(logs)
    kill = 2 * max(r["drawdown"] for r in hist) * DOLLAR

    seeds = list(range(1, args.sessions + 1))
    res = {arm: {FLOW: {v: [] for v in VARIANTS},
                 FIXED: {v: [] for v in ("none", "full overlay")}} for arm in ("calm", "stressed")}
    for sd in seeds:
        st = stress_start(sd)
        arms = {"calm": (p, None),
                "stressed": (dataclasses.replace(p, stress_start_s=st, **STRESS),
                             (st, st + STRESS_LEN_S))}
        for arm, (pp, window) in arms.items():
            for v in VARIANTS:
                res[arm][FLOW][v].append(
                    run(pp, sd, overlay(FlowSkewAS(base, model, WEIGHT), v, regimes, kill),
                        window))
            for v in ("none", "full overlay"):
                res[arm][FIXED][v].append(run(pp, sd, overlay(FixedSpread(1), v, regimes, kill),
                                              window))

    rows = [r for arm in res.values() for s in arm.values() for v in s.values() for r in v]
    checks = {
        "PnL identity in every session": all(r["identity"] for r in rows),
        f"no capped maker ever held more than {CAP} + one quote (100) shares":
            all(r.get("cap_ok", True) for r in rows),
        "after the kill switch fired, inventory only shrank":
            all(r.get("kill_ok", True) for r in rows),
    }
    fired = sum(r.get("killed", False) for r in rows)
    n = len(seeds)
    print(f"## The FYP's controller on the market maker: {n} seeds, calm and stressed\n")
    print(f"From the history block: {len(regimes.names)} regimes on 5-second bars; kill switch at "
          f"${kill / DOLLAR:,.0f} of drawdown (twice the largest of plain A–S there); cap "
          f"{CAP} shares.\n")
    print("| Check | Result | Gate |")
    print("|---|---|---|")
    for name, ok in checks.items():
        print(f"| {name} | {'yes' if ok else 'no'} | {'yes' if ok else '**NO**'} |")
        if not ok:
            failures.append(name)
    print(f"| kill switch fired | in {fired} of {len(rows)} overlay-or-not sessions | reported |")

    pol_c = res["calm"][FLOW]["policy only"]
    pol_s = res["stressed"][FLOW]["policy only"]
    fs, fc = col(pol_s, "flag_stress"), col(pol_s, "flag_calm")
    d, se = paired(fs, fc)
    lat = col(pol_s, "latency")
    print("\n## Detection (A–S + flow, policy only)\n")
    print("| Measure | Value |")
    print("|---|---:|")
    print(f"| calm sessions: bars throttled | {col(pol_c, 'throttle_calm').mean():.1%} "
          f"(FYP ceiling 5%) |")
    print(f"| calm sessions: bars paused | {col(pol_c, 'pause_calm').mean():.1%} "
          f"(FYP ceiling 1%) |")
    print(f"| stressed sessions: bars flagged during the episode | {fs.mean():.1%} ± "
          f"{mean_se(fs)[1]:.1%} |")
    print(f"| stressed sessions: bars flagged outside it | {fc.mean():.1%} ± "
          f"{mean_se(fc)[1]:.1%} |")
    print(f"| difference, paired | {d:+.1%} ± {se:.1%} (t = {d / se:+.1f}) |")
    print(f"| episodes flagged at least once | {int(np.isfinite(lat).sum())} of {n} |")
    if np.isfinite(lat).any():
        print(f"| median delay to the first flag | {np.nanmedian(lat):.0f} s |")

    for arm in ("calm", "stressed"):
        print(f"\n## {arm.capitalize()} sessions\n")
        extra = " | PnL during the episode" if arm == "stressed" else ""
        print(f"| Strategy | Overlay | Net PnL ($/hour) | vs none | Max drawdown ($) | "
              f"Max \\|inventory\\| | Volume{extra} |")
        print("|---|---|---:|---:|---:|---:|---:|" + ("---:|" if arm == "stressed" else ""))
        for strat, variants in res[arm].items():
            none = col(variants["none"], "net")
            for v, rs in variants.items():
                net = col(rs, "net")
                dd = mean_se(col(rs, "drawdown"))
                diff = "–" if v == "none" else "{:+,.0f} ± {:,.0f}".format(*paired(net, none))
                tail = ""
                if arm == "stressed":
                    tail = " | {:+,.0f} ± {:,.0f}".format(*mean_se(col(rs, "stress_pnl")))
                print(f"| {strat} | {v} | {net.mean():+,.0f} ± {mean_se(net)[1]:,.0f} | {diff} | "
                      f"{dd[0]:,.0f} ± {dd[1]:,.0f} | {col(rs, 'max_q').mean():,.0f} | "
                      f"{col(rs, 'volume').mean():,.0f}{tail} |")

    if not args.no_figure:
        plot(res, n)
        print(f"\nFigure: {FIGURE.relative_to(ROOT)}")
    print(f"\nRan in {time.time() - t0:.0f} s.")
    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for msg in failures:
        print(f"  - {msg}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
