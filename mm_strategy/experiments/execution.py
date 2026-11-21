"""Phase 4 step 2: Almgren–Chriss on our own book, against TWAP and VWAP.

Sell 30,000 shares over 56 minutes (120 child market orders, one every 28 s) in the synthetic
market of Project 3. The market is given a U-shaped intraday activity profile: arrivals are three
times as frequent at the open and close as at midday. Without that, VWAP would be TWAP.

Calibration uses separate seeds (1001–1010) and only what a trader could observe:
  σ               from the mid at 60-second sampling (step 3's estimator)
  volume profile  the market's volume in twelve 5-minute buckets (for VWAP)
  impact          probe sessions that send random-size child orders, alternating buys and
                  sells, every 28 s; cost a share against the mid just before each one,
                  regressed on size, with one slope for the whole day (classic
                  Almgren–Chriss) and with one slope per 5-minute bucket.
Then six schedules trade the same test days (seeds 1..S, paired): TWAP, VWAP, and the
Almgren–Chriss schedules at κT = 2, 4, 8 and 16, with λ chosen to give each κT.

Gate (exact, or a calibrated test):
1. Almgren–Chriss: the closed-form schedule equals the numerical minimiser of E + λV, and the
   closed-form (E, V) equal direct sums, for every λ used. λ = 0 is TWAP. With per-bucket impact
   at λ = 0, the minimiser sells n_k ∝ 1/η_k.
2. Execution: every schedule sells exactly 30,000 shares on every day. Shortfall equals
   arrival + timing + impact exactly. An executor with nothing to sell leaves the day's event
   stream (its digest) unchanged.
3. Risk: the timing term depends only on the hidden fair value, so it can be recomputed from
   the fair-value path alone; that matches the simulated term exactly on every test day. Over
   400 fair-value paths, each schedule's timing variance matches Almgren–Chriss's σ²τΣx² (true
   σ) by a two-sided χ² test at p > 6.3e-5 (4σ).
4. Cost: the fastest schedule (κT = 16) pays more impact than TWAP, paired, at 4σ.

    uv run python -m mm_strategy.experiments.execution --sessions 40   # README run (~4 min)
    uv run python -m mm_strategy.experiments.execution --sessions 6 --no-figure   # CI
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from mm_strategy import execution as ex
from mm_strategy.calibration import estimate_sigma, true_sigma
from mm_strategy.market import NS, MarketParams, fair_value_path, simulate

P = MarketParams(profile_amp=2.0)
X, START_S, END_S, N_CHILD, BUCKETS = 30_000, 120.0, 3480.0, 120, 12
TAU = (END_S - START_S) / N_CHILD
T_TOTAL = END_S - START_S
CAL_SEEDS = tuple(range(1001, 1011))
PROBE_SEED = 7
RISK_PATHS = 400
KAPPA_T = (2, 4, 8, 16)
P_4SIGMA = 6.3e-5
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "p4_execution.png"
CHILD_TIMES = START_S + TAU * np.arange(N_CHILD)


@dataclass
class Calibration:
    sigma: object  # calibration.Estimate, ticks/√s
    profile: np.ndarray  # share of window volume per bucket
    classic: ex.ImpactModel
    bucket: ex.BucketImpact
    probe_n: np.ndarray
    probe_cost: np.ndarray
    probe_t: np.ndarray


def calibrate() -> Calibration:
    plain = [simulate(P, sd) for sd in CAL_SEEDS]
    sigma = estimate_sigma(plain, 60.0)
    profile = ex.volume_profile(plain, START_S, END_S, BUCKETS)
    rng = np.random.default_rng(PROBE_SEED)
    ts, ns, cs = [], [], []
    for sd in CAL_SEEDS:
        probe = ex.Executor(START_S, TAU, rng.integers(1, 31, N_CHILD) * 50,
                            sides=np.where(np.arange(N_CHILD) % 2 == 0, -1, 1), carry=False)
        simulate(P, sd, probe)
        t, n, c = ex.probe_costs(probe)
        ts.append(t)
        ns.append(n)
        cs.append(c / P.tick)
    t, n, c = np.concatenate(ts), np.concatenate(ns), np.concatenate(cs)
    return Calibration(sigma, profile, ex.fit_impact(n, c),
                       ex.fit_bucket_impact(t, n, c, START_S, END_S, BUCKETS), n, c, t)


def schedules(cal: Calibration) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    sig, eta = cal.sigma.value, cal.classic.eta
    out = {"TWAP": ex.twap(X, N_CHILD),
           "VWAP": ex.vwap(X, cal.profile[ex.bucket_of(CHILD_TIMES, START_S, END_S, BUCKETS)])}
    lams = {"TWAP": 0.0, "VWAP": float("nan")}
    for kt in KAPPA_T:
        lam = ex.lam_for_kappa(kt / T_TOTAL, sig, TAU, eta)
        name = f"A–C κT={kt}"
        out[name] = ex.ac_holdings(X, N_CHILD, TAU, sig, eta, lam)
        lams[name] = lam
    return out, lams


def check_theory(cal: Calibration, lams: dict[str, float]) -> bool:
    sig, model = cal.sigma.value, cal.classic
    worst_x = worst_ev = 0.0
    for lam in (v for v in lams.values() if np.isfinite(v)):
        closed = ex.ac_holdings(X, N_CHILD, TAU, sig, model.eta, lam)
        numeric = ex.optimal_holdings(X, np.full(N_CHILD, model.eta), sig, TAU, lam)
        worst_x = max(worst_x, float(np.max(np.abs(closed - numeric))))
        e, v = ex.frontier_point(X, N_CHILD, TAU, sig, model, lam)
        worst_ev = max(worst_ev, abs(e / ex.expected_cost(closed, model) - 1),
                       abs(v / ex.variance(closed, sig, TAU) - 1))
    twap_ok = np.allclose(ex.ac_holdings(X, N_CHILD, TAU, sig, model.eta, 0.0),
                          ex.twap(X, N_CHILD), rtol=0, atol=1e-9)
    eta_k = cal.bucket.per_child(CHILD_TIMES, START_S, END_S)
    n0 = -np.diff(ex.optimal_holdings(X, eta_k, sig, TAU, 0.0))
    inv_ok = np.allclose(n0, X * (1 / eta_k) / np.sum(1 / eta_k), rtol=1e-9, atol=1e-9)
    print(f"\nTheory: closed form vs numerical minimiser, max |Δx| = {worst_x:.1e} shares; "
          f"closed-form (E, V) vs sums, max relative error {worst_ev:.1e}; λ = 0 is TWAP: "
          f"{'yes' if twap_ok else 'NO'}; per-bucket impact at λ = 0 sells n ∝ 1/η: "
          f"{'yes' if inv_ok else 'NO'}")
    return worst_x < 1e-6 and worst_ev < 1e-9 and twap_ok and inv_ok


def execute(sizes: dict[str, np.ndarray], sessions: int) -> tuple[dict, bool]:
    res: dict[str, list] = {k: [] for k in sizes}
    ok = True
    for sd in range(1, sessions + 1):
        for name, sz in sizes.items():
            e = ex.Executor(START_S, TAU, sz)
            s = simulate(P, sd, e)
            sf = ex.shortfall(s, e, START_S)
            parts = sf.arrival + sf.timing + sf.impact
            ok &= sf.filled == X and abs(sf.total - parts) < 1e-9
            f = e.fill_arrays()
            res[name].append({"total": sf.total, "impact": sf.impact, "timing": sf.timing,
                              "vs_mid": sf.vs_mid, "fill_ts": f["ts"], "fill_qty": f["qty"],
                              "short_children": sum(c[2] < c[1] for c in e.children)})
    idle = ex.Executor(START_S, TAU, np.zeros(N_CHILD, dtype=int))
    with_idle, plain = simulate(P, 1, idle), simulate(P, 1)
    ok &= with_idle.digest == plain.digest and not idle.fills
    print(f"\nExecution: every schedule filled {X:,} shares on all {sessions} days and the "
          f"shortfall split adds up: {'yes' if ok else 'NO'} (an idle executor leaves the "
          f"digest at {plain.digest:#018x})")
    return res, ok


def fair_value_alone(seed: int) -> np.ndarray:
    """The session's fair-value path, regenerated from its seed without running the market."""
    return fair_value_path(P, np.random.default_rng(np.random.SeedSequence(seed).spawn(5)[0]))


def timing_term(fv: np.ndarray, ts_ns: np.ndarray, qty: np.ndarray) -> float:
    """Σ q (V0 − V_t) / X in ticks a share, V0 at the first child's time."""
    step = P.fv_step_ms * 1_000_000
    v = fv[np.minimum(np.asarray(ts_ns, dtype=np.int64) // step, len(fv) - 1)]
    v0 = fv[int(round(START_S * NS)) // step]
    return float(np.sum(np.asarray(qty, dtype=float) * (v0 - v)) / X / P.tick)


def check_risk(hold: dict[str, np.ndarray], sizes: dict[str, np.ndarray],
               res: dict) -> tuple[bool, dict[str, float]]:
    days = len(res["TWAP"])
    exact = True
    for sd in range(1, days + 1):
        fv = fair_value_alone(sd)
        for k in sizes:
            r = res[k][sd - 1]
            exact &= abs(timing_term(fv, r["fill_ts"], r["fill_qty"]) - r["timing"]) < 1e-9
    child_ts = np.rint(CHILD_TIMES * NS).astype(np.int64)
    paths = []
    for sd in range(1, RISK_PATHS + 1):
        fv = fair_value_alone(sd)
        paths.append({k: timing_term(fv, child_ts, n) for k, n in sizes.items()})
    sig_t = true_sigma(P)
    ok, sd_paths = exact, {}
    short = {k: sum(r["short_children"] > 0 for r in res[k]) for k in sizes}
    print(f"\nRisk: the timing term recomputed from each day's fills and the fair-value path "
          f"alone matches the simulation on all {days} days: {'yes' if exact else 'NO'}")
    print("  days on which a child order ran out of bids and rolled the rest forward: "
          + ", ".join(f"{k} {v}" for k, v in short.items()))
    print(f"  over {RISK_PATHS} fair-value paths (seeds 1–{RISK_PATHS}), s.d. a share vs "
          f"Almgren–Chriss's √V/X at the true σ:")
    for k, h in hold.items():
        x = np.array([p_[k] for p_ in paths])
        pred = ex.variance(h, sig_t, TAU) / X**2
        chi = float(np.sum(x**2) / pred)
        p = 2 * min(stats.chi2.cdf(chi, len(x)), stats.chi2.sf(chi, len(x)))
        ok &= p > P_4SIGMA
        sd_paths[k] = float(x.std(ddof=1))
        print(f"  {k:<12} {sd_paths[k]:5.1f} vs {np.sqrt(pred):5.1f}  (χ² p = {p:.2f})")
    return ok, sd_paths


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def report(cal: Calibration, hold: dict[str, np.ndarray], lams: dict[str, float],
           sizes: dict[str, np.ndarray], res: dict) -> bool:
    sig_t = true_sigma(P)
    c, b = cal.classic, cal.bucket
    print(f"\nCalibration (seeds {CAL_SEEDS[0]}–{CAL_SEEDS[-1]}, observable data only)")
    print(f"  σ = {cal.sigma.value:.3f} ± {cal.sigma.se:.3f} ticks/√s (true {sig_t:.3f})")
    print("  volume per 5 min, relative to the mean: "
          + " ".join(f"{v:.2f}" for v in cal.profile * BUCKETS))
    print(f"  one slope:       cost = {c.epsilon:.3f} + {c.eta * 1e3:.3f}e-3·n ticks a share, "
          f"R² {c.r2:.3f} ({len(cal.probe_n):,} probe orders)")
    print(f"  slope per 5 min: cost = {b.epsilon:.3f} + η_b·n, η_b = "
          + " ".join(f"{v * 1e3:.2f}" for v in b.eta) + f" (e-3), R² {b.r2:.3f}")

    eta_k = b.per_child(CHILD_TIMES, START_S, END_S)
    print(f"\n{'schedule':<12}{'λ ($⁻¹)':>10}{'1st child':>10}{'predicted cost':>16}"
          f"{'impact':>15}{'vs mid':>8}{'risk':>7}{'timing s.d.':>13}{'shortfall':>17}{'s.d.':>7}")
    print(f"{'':<12}{'':>10}{'':>10}{'1 slope / 12':>16}{'(realized)':>15}{'':>8}"
          f"{'(A–C)':>7}{'(realized)':>13}{'mean ± s.e.':>17}{'':>7}")
    for name, h in hold.items():
        rows = res[name]
        imp, tot, tim = col(rows, "impact"), col(rows, "total"), col(rows, "timing")
        pc, pb = ex.expected_cost(h, c) / X, b.expected_cost(h, eta_k) / X
        risk = np.sqrt(ex.variance(h, sig_t, TAU)) / X
        lam = lams[name]
        lam_s = "—" if not np.isfinite(lam) else f"{100 * lam:.1e}" if lam else "0"
        m, se = mean_se(imp)
        mt, set_ = mean_se(tot)
        print(f"{name:<12}{lam_s:>10}{sizes[name][0]:>10,}{pc:>8.2f} / {pb:<5.2f}"
              f"{m:>9.3f} ± {se:.3f}{col(rows, 'vs_mid').mean():>8.3f}{risk:>7.1f}"
              f"{tim.std(ddof=1):>13.1f}{mt:>+10.2f} ± {set_:<5.2f}{tot.std(ddof=1):>7.1f}")
    print("  ticks a share = bp of the $100 arrival price. Impact: fills against the hidden fair "
          "value. Vs mid: fills against the mid before each child (what a desk sees). Risk: "
          "A–C's √V/X at the true σ.")

    print("\nPaired against TWAP, same days")
    base = res["TWAP"]
    for name in hold:
        if name == "TWAP":
            continue
        d_imp = col(res[name], "impact") - col(base, "impact")
        d_tot = col(res[name], "total") - col(base, "total")
        m, se = mean_se(d_imp)
        mt, set_ = mean_se(d_tot)
        print(f"  {name:<12} impact {m:>+7.3f} ± {se:.3f} (t {m / se:>+5.1f})   "
              f"shortfall {mt:>+6.2f} ± {set_:.2f}")
    fast = f"A–C κT={KAPPA_T[-1]}"
    d = col(res[fast], "impact") - col(base, "impact")
    m, se = mean_se(d)
    return m - 4 * se > 0


def plot(cal: Calibration, hold: dict[str, np.ndarray], res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    sig, sig_t = cal.sigma.value, true_sigma(P)
    c, b = cal.classic, cal.bucket
    eta_k = b.per_child(CHILD_TIMES, START_S, END_S)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=150)
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

    colors = {"TWAP": ink2, "VWAP": palette[3]}
    shades = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(KAPPA_T)))
    for kt, sh in zip(KAPPA_T, shades, strict=True):
        colors[f"A–C κT={kt}"] = sh

    # Efficient frontier: model curves and realized points.
    ax = axes[0, 0]
    kts = np.r_[0.0, np.geomspace(0.05, 40, 80)]
    lam_grid = [ex.lam_for_kappa(k / T_TOTAL, sig, TAU, c.eta) for k in kts]
    one = np.array([(np.sqrt(ex.variance(h, sig_t, TAU)) / X, ex.expected_cost(h, c) / X)
                    for h in (ex.ac_holdings(X, N_CHILD, TAU, sig, c.eta, la)
                              for la in lam_grid)])
    per = np.array([(np.sqrt(ex.variance(h, sig_t, TAU)) / X, b.expected_cost(h, eta_k) / X)
                    for h in (ex.optimal_holdings(X, eta_k, sig, TAU, la) for la in lam_grid)])
    ax.plot(one[:, 0], one[:, 1], color=ink2, lw=1.2, ls="--",
            label="A–C frontier, one impact slope")
    ax.plot(per[:, 0], per[:, 1], color=palette[0], lw=1.4,
            label="A–C frontier, impact per 5 min")
    offsets = {"TWAP": (4, 6), "VWAP": (4, -14), "A–C κT=2": (2, -15), "A–C κT=4": (4, -15)}
    for name, h in hold.items():
        imp = col(res[name], "impact")
        risk = np.sqrt(ex.variance(h, sig_t, TAU)) / X
        ax.errorbar(risk, imp.mean(), yerr=1.96 * imp.std(ddof=1) / np.sqrt(len(imp)),
                    fmt="o", ms=5, color=colors[name], capsize=2, lw=1)
        dx, dy = offsets.get(name, (6, 4))
        ax.annotate(name.replace("A–C ", ""), (risk, imp.mean()), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=ink2,
                    ha="right" if dx < 0 else "left")
    ax.errorbar([], [], yerr=[], fmt="o", ms=5, color=ink2, capsize=2, lw=1,
                label="realized impact, 95% CI")
    ax.set_xlim(0, max(one[:, 0].max(), 1) * 1.12)
    ax.set_ylim(0, 6)
    style(ax, "Selling faster buys less risk with more impact",
          "Risk: s.d. of shortfall from price moves (bp), Almgren–Chriss at the true σ",
          "Cost: impact (bp)")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper right")

    # The schedules.
    ax = axes[0, 1]
    t_min = np.r_[CHILD_TIMES, END_S] / 60
    for name, h in hold.items():
        ax.step(t_min, h / 1000, where="post", color=colors[name], lw=1.5, label=name)
    style(ax, "The schedules: shares left to sell", "Minutes from the open",
          "Shares left (thousands)")
    ax.legend(frameon=False, fontsize=7, labelcolor=ink, loc="upper right")

    # Liquidity through the day.
    ax = axes[1, 0]
    mid_b = START_S / 60 + (np.arange(BUCKETS) + 0.5) * T_TOTAL / BUCKETS / 60
    ax.bar(mid_b, b.eta * 1e3, width=T_TOTAL / BUCKETS / 60 * 0.8, color=palette[0],
           label="impact slope η per 5 min (left)")
    ax.axhline(c.eta * 1e3, color=ink2, lw=1.1, ls="--", label="one slope for the day")
    ax2 = ax.twinx()
    ax2.plot(mid_b, cal.profile * BUCKETS, color=palette[1], marker="o", ms=4, lw=1.5,
             label="market volume (right)")
    ax2.set_ylim(0, cal.profile.max() * BUCKETS * 1.5)
    ax.set_ylim(0, b.eta.max() * 1e3 * 1.5)
    ax2.tick_params(colors=ink2, labelsize=8)
    ax2.set_ylabel("Volume relative to the mean", color=ink2, fontsize=8.5)
    for sp in ("top",):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(grid)
    ratio = max(b.eta[0], b.eta[-1]) / b.eta.max()
    style(ax, f"Impact at the open and close is ≤{ratio:.0%} of its midday peak",
          "Minutes from the open",
          "Impact slope (1e-3 ticks a share per share)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, labelcolor=ink, loc="upper left")

    # Predicted vs realized cost.
    ax = axes[1, 1]
    lim = 0.0
    for name, h in hold.items():
        imp = col(res[name], "impact")
        m, se = mean_se(imp)
        for pred, mk, fill in ((ex.expected_cost(h, c) / X, "s", surface),
                               (b.expected_cost(h, eta_k) / X, "o", colors[name])):
            ax.errorbar(pred, m, yerr=1.96 * se, fmt=mk, ms=6, mfc=fill, mec=colors[name],
                        color=colors[name], capsize=2, lw=1)
            lim = max(lim, pred, m)
    ax.plot([0, lim * 1.1], [0, lim * 1.1], color=ink, lw=0.9)
    ax.plot([], [], "s", mfc=surface, mec=ink2, label="predicted with one slope")
    ax.plot([], [], "o", color=ink2, label="predicted with a slope per 5 min")
    ax.set_xlim(0, lim * 1.1)
    ax.set_ylim(0, lim * 1.1)
    style(ax, "Time-of-day impact predicts the realized cost", "Predicted impact (bp)",
          "Realized impact (bp), 95% CI")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=ink, loc="upper left")

    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sessions", type=int, default=40)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    print(f"Sell {X:,} shares from minute {START_S / 60:g} to {END_S / 60:g} in {N_CHILD} child "
          f"market orders (every {TAU:g} s); activity profile 1 + {P.profile_amp:g}·(2t/T − 1)²; "
          f"{args.sessions} test days")
    cal = calibrate()
    hold, lams = schedules(cal)
    sizes = {k: ex.child_sizes(h) for k, h in hold.items()}
    ok_theory = check_theory(cal, lams)
    res, ok_exec = execute(sizes, args.sessions)
    ok_cost = report(cal, hold, lams, sizes, res)
    ok_risk, _ = check_risk(hold, sizes, res)
    if not args.no_figure:
        plot(cal, hold, res)
        print(f"\nFigure written to {FIGURE}")
    ok = {"theory": ok_theory, "execution": ok_exec, "risk": ok_risk, "cost": ok_cost}
    print("\nPhase 4 step 2 gate: " + " · ".join(f"{k} {'PASS' if v else 'FAIL'}"
                                                 for k, v in ok.items()))
    if not all(ok.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
