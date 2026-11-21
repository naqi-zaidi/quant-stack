"""Project 3 step 6 on the AAPL day: the risk overlay in step 5's walk-forward (local only).

Same blocks, replay and walk-forward as experiments.aapl. From the 9:30-10:30 history block
come the k-means regimes (5-second bars of the mid) and the kill switch's threshold: twice the
largest drawdown plain A–S had in a history block, with σ and k fitted on that block. The cap
is 500 shares, as on the synthetic market. Each test block's overlay warms its features and
detectors up on the previous block's mids, so it can act from the block's first bar.

    uv run python -m mm_strategy.experiments.risk --aapl
"""

from __future__ import annotations

import time

import numpy as np

from mm_strategy.calibration import estimate_intensity, estimate_sigma
from mm_strategy.experiments.aapl import (
    DATA,
    DELTAS,
    HISTORY,
    N_BLOCKS,
    ROOT,
    TICK,
    Day,
    FeatureLogger,
    block_bounds,
)
from mm_strategy.experiments.avellaneda import mean_se, paired
from mm_strategy.experiments.flow import GAMMA, HORIZON_S, WEIGHT
from mm_strategy.experiments.risk import BAR_S, CAP, DOLLAR, regimes_from
from mm_strategy.maker import AvellanedaStoikov
from mm_strategy.replay import LobsterSource, ReplayMaker, TapeReplay, mid_at
from mm_strategy.risk import RiskOverlay
from mm_strategy.signals import FlowSkewAS, fit

NS = 1_000_000_000
NAMES = ("A–S", "A–S + overlay", "A–S + flow", "A–S + flow + overlay")


def drawdown(rm: ReplayMaker, quotes) -> float:
    """Max drawdown of the marked PnL on 1-second samples, dollars."""
    s = np.array(rm.samples, dtype=float)[::10]
    pnl = s[:, 2] + s[:, 1] * mid_at(quotes, s[:, 0].astype(np.int64), TICK) * TICK
    pnl = pnl[np.isfinite(pnl)]
    return float((np.maximum.accumulate(pnl) - pnl).max()) / DOLLAR if len(pnl) else 0.0


def main(no_figure: bool = True) -> None:
    if not DATA.exists():
        raise SystemExit(f"{DATA.relative_to(ROOT)} not found: see data/README.md")
    t0 = time.time()
    day = Day(DATA)
    windows = [day.block(i) for i in range(N_BLOCKS)]
    rp = TapeReplay(LobsterSource(str(DATA)))
    sig_h = estimate_sigma(windows[:HISTORY], 60.0).value
    k_h = estimate_intensity(windows[:HISTORY], DELTAS).k.value
    hist_logs, hist_dd = [], []
    for i in range(HISTORY):
        start, end = block_bounds(i)
        logger = FeatureLogger()
        plain = ReplayMaker(AvellanedaStoikov(GAMMA, sig_h, k_h), start, end)
        rp.run([ReplayMaker(logger, start, end), plain], end)
        hist_logs.append(logger.features.log)
        hist_dd.append(drawdown(plain, rp.quotes()))
    regimes = regimes_from(hist_logs)
    kill = 2 * max(hist_dd) * DOLLAR

    logs = list(hist_logs)
    res = {k: [] for k in NAMES}
    checks = {"identity": True, "cap": True, "kill": True}
    acts = {"throttle": 0, "pause": 0, "allow": 0}
    fired = 0
    for i in range(HISTORY, N_BLOCKS):
        start, end = block_bounds(i)
        base = AvellanedaStoikov(GAMMA, estimate_sigma(windows[:i], 60.0).value,
                                 estimate_intensity(windows[:i], DELTAS).k.value)
        model = fit(logs, HORIZON_S)
        warm = np.asarray(logs[-1], float)[:, 1]

        def guarded(inner, warm=warm):
            o = RiskOverlay(inner, regimes, bar_s=BAR_S, inventory_cap=CAP, max_drawdown=kill)
            o.warm_up(warm)
            return o

        strategies = {"A–S": base, "A–S + overlay": guarded(base),
                      "A–S + flow": FlowSkewAS(base, model, WEIGHT),
                      "A–S + flow + overlay": guarded(FlowSkewAS(base, model, WEIGHT))}
        makers = {k: ReplayMaker(s, start, end) for k, s in strategies.items()}
        logger = FeatureLogger()
        rp.run([*makers.values(), ReplayMaker(logger, start, end)], end)
        logs.append(logger.features.log)
        q = rp.quotes()
        for k, rm in makers.items():
            pnl = rm.pnl()
            inv = np.array([x[1] for x in rm.samples])
            checks["identity"] &= pnl.identity_holds
            res[k].append({"net": pnl.net / DOLLAR, "dd": drawdown(rm, q), "volume": pnl.volume,
                           "max_q": int(np.abs(inv).max())})
            st = rm.strategy
            if isinstance(st, RiskOverlay):
                checks["cap"] &= int(np.abs(inv).max()) <= CAP + 100
                for b in st.bars:
                    acts[b.action] += 1
                if st.killed_at is not None:
                    fired += 1
                    s = np.array(rm.samples, dtype=float)
                    checks["kill"] &= bool(np.all(np.diff(np.abs(s[s[:, 0] >= st.killed_at, 1]))
                                                  <= 0))

    n = N_BLOCKS - HISTORY
    tot = sum(acts.values())
    print(f"## The overlay on AAPL: {n} half-hour blocks, 10:30–16:00, walk-forward\n")
    print(f"From the history block: kill switch at ${kill / DOLLAR:,.0f} of drawdown (twice the "
          f"largest of plain A–S there); cap {CAP} shares; regimes on 5-second bars.\n")
    print("| Check | Result | Gate |")
    print("|---|---|---|")
    for name, key in (("replayed book matches LOBSTER after every message", None),
                      ("PnL identity, every maker in every block", "identity"),
                      (f"no capped maker held more than {CAP} + 100 shares", "cap"),
                      ("after the kill switch fired, inventory only shrank", "kill")):
        ok = (rp.rp.stats.mismatches == 0 and rp.rp.stats.errors == 0) if key is None \
            else checks[key]
        print(f"| {name} | {'yes' if ok else 'no'} | {'yes' if ok else '**NO**'} |")
        if not ok:
            raise SystemExit(f"Gate: FAIL ({name})")
    print(f"| kill switch fired | in {fired} of {2 * n} overlay blocks | reported |")
    print(f"| overlay's bars: throttled / paused | {acts['throttle'] / tot:.1%} / "
          f"{acts['pause'] / tot:.1%} | reported |")

    print("\n| Strategy | Net PnL ($/hour) | Worst block ($) | Max drawdown in a block ($) | "
          "Max \\|inventory\\| | Volume (shares/hour) |")
    print("|---|---:|---:|---:|---:|---:|")
    for k in NAMES:
        net = 2 * np.array([r["net"] for r in res[k]])
        dd = mean_se([r["dd"] for r in res[k]])
        print(f"| {k} | {net.mean():+,.0f} ± {mean_se(net)[1]:,.0f} | "
              f"{min(r['net'] for r in res[k]):+,.0f} | {dd[0]:,.0f} ± {dd[1]:,.0f} | "
              f"{max(r['max_q'] for r in res[k]):,} | "
              f"{2 * np.mean([r['volume'] for r in res[k]]):,.0f} |")
    print("\n| Paired across blocks ($/hour) | Difference | t |")
    print("|---|---:|---:|")
    for a, b in (("A–S + overlay", "A–S"), ("A–S + flow + overlay", "A–S + flow")):
        d, se = paired(2 * np.array([r["net"] for r in res[a]]),
                       2 * np.array([r["net"] for r in res[b]]))
        print(f"| {a} − {b} | {d:+,.0f} ± {se:,.0f} | {d / se:+.1f} |")
    print(f"\nRan in {time.time() - t0:.0f} s.")
    print("\nGate: PASS")
