"""Phase 4 step 1: discrete delta hedging of a short call, swept over hedge frequency and costs.

A dealer sells a 3-month at-the-money call at 20% implied vol and delta-hedges it n times over its
life. The stock's realized vol is 15%, 20% or 25%. The hedge uses either the implied vol or the
realized vol. Every policy and every n runs on the same 20,000 paths. Paths at different vols reuse
the same normals, so the comparisons are common-random-number pairs.

Gate (exact or asymptotic, nothing tuned):

1. Martingale identity. Mean P&L is BS(20%) − BS(sigma_r) within 4 s.e. in every cell: 5 policies
   × 9 frequencies. Exact for any hedge, because the stock drifts at r.
2. Convergence. P&L minus its continuous-time limit, BS(sigma_i) − BS(sigma_h) +
   (sigma_h² − sigma_r²)·dollar gamma, has s.d. falling like 1/√n. The log-log slope over
   n ≥ 64 is within 0.05 of −½ for every policy.
3. Derman–Kamal. Hedged at realized vol, the P&L s.d. is within 5% of sqrt(π/(4n))·vega·sigma_r
   for n ≥ 64.
4. Gamma P&L. The second-order decomposition explains a fraction of P&L variance that rises with
   n, and is at least 98% at n = 1024.
5. Leland scaling. Rebalancing turnover grows like √n: log-log slope within 0.05 of ½ over n ≥ 64.

Costs are reported, not gated: the RMSE of net P&L around its frictionless target, per cost
level, and the frequency that minimizes it.

    uv run python -m mc_engine.experiments.hedging          # tables, gate, figure (~15 s)
    uv run python -m mc_engine.experiments.hedging --no-figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mc_engine.black_scholes import bs_price
from mc_engine.hedging import HedgeRun, delta_hedge, derman_kamal_sd, gbm_grid

S0, K, T, R, SIGMA_I = 100.0, 100.0, 0.25, 0.03, 0.20
SIGMA_R = (0.15, 0.20, 0.25)
N_STEPS = 1024
N_GRID = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
N_PATHS = 20_000
SEED = 20261121
FIT_FROM = 64
COSTS_BP = (2, 5, 10)
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "p4_hedging.png"

Policy = tuple[float, str]  # (sigma_r, "implied" | "realized"); at sigma_r = sigma_i they coincide


def policies() -> list[Policy]:
    out: list[Policy] = []
    for sr in SIGMA_R:
        out.append((sr, "implied"))
        if sr != SIGMA_I:
            out.append((sr, "realized"))
    return out


def label(p: Policy) -> str:
    sr, how = p
    return f"σr {sr:.0%}, hedge at {'both' if sr == SIGMA_I else how}"


def hedge_vol(p: Policy) -> float:
    return SIGMA_I if p[1] == "implied" else p[0]


def run(n_paths: int) -> dict[Policy, dict[int, HedgeRun]]:
    res: dict[Policy, dict[int, HedgeRun]] = {}
    for sr in SIGMA_R:
        paths = gbm_grid(S0, R, sr, T, N_STEPS, n_paths, SEED)
        for p in (q for q in policies() if q[0] == sr):
            res[p] = {n: delta_hedge(paths, K, T, R, SIGMA_I, hedge_vol(p), n) for n in N_GRID}
    return res


def limit(p: Policy, h: HedgeRun) -> np.ndarray:
    """Continuous-time limit of the P&L, path by path."""
    return h.locked_in + (hedge_vol(p) ** 2 - p[0] ** 2) * h.dollar_gamma


def target(sr: float) -> float:
    return float(bs_price(S0, K, T, R, SIGMA_I) - bs_price(S0, K, T, R, sr))


def r2_gamma(h: HedgeRun) -> float:
    return float(1.0 - np.var(h.residual) / np.var(h.pnl))


def slope(n: np.ndarray, y: np.ndarray) -> float:
    sel = n >= FIT_FROM
    return float(np.polyfit(np.log(n[sel]), np.log(y[sel]), 1)[0])


def gate(res: dict[Policy, dict[int, HedgeRun]]) -> dict[str, bool]:
    n = np.array(N_GRID, dtype=float)
    ok = dict.fromkeys(("martingale", "convergence", "derman-kamal", "gamma", "leland"), True)
    print(f"\n{'policy':<28}{'n':>6}{'mean ± s.e.':>18}{'target':>9}{'z':>7}{'s.d.':>8}"
          f"{'D–K':>8}{'s.d.(P&L−lim)':>15}{'R² γ':>8}")
    for p, runs in res.items():
        tgt = target(p[0])
        for i, (nn, h) in enumerate(runs.items()):
            m, se = h.mean_se()
            z = (m - tgt) / se
            sd = float(h.pnl.std(ddof=1))
            dk = derman_kamal_sd(S0, K, T, R, p[0], nn)
            gap = float(np.std(h.pnl - limit(p, h), ddof=1))
            print(f"{label(p) if i == 0 else '':<28}{nn:>6}{m:>+11.4f} ± {se:.4f}{tgt:>+9.4f}"
                  f"{z:>+7.2f}{sd:>8.4f}{dk:>8.4f}{gap:>15.4f}{r2_gamma(h):>8.4f}")
            ok["martingale"] &= abs(z) < 4
            if p[1] == "realized" or p[0] == SIGMA_I:
                ok["derman-kamal"] &= nn < FIT_FROM or abs(sd / dk - 1) < 0.05
        gaps = np.array([np.std(h.pnl - limit(p, h), ddof=1) for h in runs.values()])
        r2 = np.array([r2_gamma(h) for h in runs.values()])
        turn = np.array([h.rebalance_traded.mean() for h in runs.values()])
        s_gap, s_turn = slope(n, gaps), slope(n, turn)
        print(f"{'':<28}slope of s.d.(P&L − limit) {s_gap:+.3f} · slope of turnover "
              f"{s_turn:+.3f} (fit over n ≥ {FIT_FROM})")
        ok["convergence"] &= abs(s_gap + 0.5) < 0.05
        ok["gamma"] &= bool(np.all(np.diff(r2) > 0) and r2[-1] >= 0.98)
        ok["leland"] &= abs(s_turn - 0.5) < 0.05
    return ok


def report_limits(res: dict[Policy, dict[int, HedgeRun]]) -> None:
    print("\nWhat rebalancing can't remove: s.d. of the continuous-time limit")
    print(f"{'policy':<28}{'s.d. at n=1024':>16}{'s.d. of limit':>15}{'corr(P&L, limit)':>18}")
    for p, runs in res.items():
        h = runs[N_GRID[-1]]
        lim = limit(p, h)
        sd_lim = float(np.std(lim, ddof=1))
        corr = f"{np.corrcoef(h.pnl, lim)[0, 1]:.4f}" if sd_lim > 0 else "—"
        print(f"{label(p):<28}{h.pnl.std(ddof=1):>16.4f}{sd_lim:>15.4f}{corr:>18}")


def cost_table(res: dict[Policy, dict[int, HedgeRun]]) -> dict[int, tuple[np.ndarray, int]]:
    """RMSE of net P&L around the frictionless target, sigma_r = sigma_i (target 0)."""
    runs = res[(SIGMA_I, "implied")]
    out: dict[int, tuple[np.ndarray, int]] = {}
    print("\nCosts, σr = σi = 20%: mean cost and RMSE of net P&L around 0, per call "
          f"(premium {bs_price(S0, K, T, R, SIGMA_I):.3f})")
    head = "".join(f"{f'{bp} bp cost':>12}{'RMSE':>9}" for bp in COSTS_BP)
    print(f"{'n':>6}{'s.d.':>9}{head}")
    table = {bp: np.array([np.sqrt(np.mean(h.net(bp * 1e-4) ** 2)) for h in runs.values()])
             for bp in COSTS_BP}
    for bp, rmse in table.items():
        out[bp] = (rmse, N_GRID[int(np.argmin(rmse))])
    for i, (n, h) in enumerate(runs.items()):
        cells = "".join(
            f"{h.costs(bp * 1e-4).mean():>12.4f}{table[bp][i]:>8.4f}"
            f"{'*' if n == out[bp][1] else ' '}" for bp in COSTS_BP)
        print(f"{n:>6}{h.pnl.std(ddof=1):>9.4f}{cells}")
    print("  * = RMSE-minimizing n on the grid. "
          + " · ".join(f"{bp} bp → n* = {out[bp][1]}" for bp in COSTS_BP))
    return out


def plot(res: dict[Policy, dict[int, HedgeRun]], best: dict[int, tuple[np.ndarray, int]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    color = {0.15: palette[0], 0.20: palette[2], 0.25: palette[1]}
    n = np.array(N_GRID, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), dpi=150)
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

    def log_axes(ax, yticks):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(N_GRID, [str(x) for x in N_GRID])
        ax.set_yticks(yticks, [f"{y:g}" for y in yticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax = axes[0]
    for p, runs in res.items():
        sd = np.array([h.pnl.std(ddof=1) for h in runs.values()])
        c = color[p[0]]
        if p[1] == "realized" or p[0] == SIGMA_I:
            name = "implied = realized" if p[0] == SIGMA_I else "realized-vol hedge"
            ax.plot(n, sd, color=c, marker="o", ms=4, lw=1.6, label=f"σr {p[0]:.0%}, {name}")
            ax.plot(n, [derman_kamal_sd(S0, K, T, R, p[0], int(x)) for x in n], color=ink,
                    lw=0.8, ls=(0, (3, 3)))
        else:
            ax.plot(n, sd, color=c, marker="o", ms=4, mfc=surface, lw=1.3, ls="--",
                    label=f"σr {p[0]:.0%}, implied-vol hedge")
            ax.axhline(np.std(limit(p, runs[N_GRID[-1]]), ddof=1), color=c, lw=0.9, ls=":")
    ax.plot([], [], color=ink, lw=0.8, ls=(0, (3, 3)), label="Derman–Kamal √(π/4n)·vega·σr")
    ax.plot([], [], color=ink2, lw=0.9, ls=":", label="s.d. of the n → ∞ limit")
    log_axes(ax, (0.1, 0.2, 0.5, 1, 2))
    style(ax, "Only the realized-vol hedge converges to zero risk",
          "Rebalances over 3 months (n)", "P&L standard deviation (per call)")
    ax.legend(frameon=False, fontsize=7, labelcolor=ink, loc="lower left")

    ax = axes[1]
    for sr in (0.15, 0.25):
        h = res[(sr, "implied")][N_GRID[-1]]
        x, y = limit((sr, "implied"), h), h.pnl
        idx = np.random.default_rng(0).choice(len(x), 2500, replace=False)
        ax.scatter(x[idx], y[idx], s=4, alpha=0.35, color=color[sr], lw=0,
                   label=f"σr {sr:.0%}: corr {np.corrcoef(x, y)[0, 1]:.3f}")
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color=ink, lw=0.9)
    ax.set_xlim(lo, hi)
    style(ax, "Implied-vol hedge: P&L follows the gamma term",
          r"$(\sigma_i^2-\sigma_r^2)\,\sum\ \frac{1}{2}\Gamma S^2\,dt$ along the path",
          f"Hedged P&L, n = {N_GRID[-1]}")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper left", markerscale=3)

    ax = axes[2]
    runs = res[(SIGMA_I, "implied")]
    sd = np.array([h.pnl.std(ddof=1) for h in runs.values()])
    ax.plot(n, sd, color=ink2, lw=1.3, marker="o", ms=3.5, label="no costs")
    for j, bp in enumerate(COSTS_BP):
        rmse, n_star = best[bp]
        c = (palette[0], palette[3], palette[1])[j]
        ax.plot(n, rmse, color=c, lw=1.6, marker="o", ms=3.5, label=f"{bp} bp per trade")
        ax.plot([n_star], [rmse[N_GRID.index(n_star)]], marker="o", ms=9, mfc="none", mec=c,
                mew=1.6)
    log_axes(ax, (0.1, 0.2, 0.5, 1, 1.5))
    style(ax, "With costs, hedging more often stops paying",
          "Rebalances over 3 months (n)", "RMSE of net P&L around 0 (σi = σr = 20%)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper right")
    ax.annotate("circled: best n", xy=(0.03, 0.04), xycoords="axes fraction", fontsize=7.5,
                color=ink2)

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", type=int, default=N_PATHS)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    print(f"Short {T:g}-year call, S0 = K = {S0:g}, r = {R:.0%}, sold at σi = {SIGMA_I:.0%}; "
          f"{args.paths:,} paths, {N_STEPS} steps")
    res = run(args.paths)
    ok = gate(res)
    report_limits(res)
    best = cost_table(res)
    if not args.no_figure:
        plot(res, best)
        print(f"\nFigure written to {FIGURE}")
    print("\nPhase 4 step 1 gate: " + " · ".join(f"{k} {'PASS' if v else 'FAIL'}"
                                                 for k, v in ok.items()))
    if not all(ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
