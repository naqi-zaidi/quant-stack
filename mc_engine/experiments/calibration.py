"""Project 1 step 6: calibrate Heston to a real SPY implied-volatility snapshot.

Data: data/snapshots/spy_options_2026-11-19.csv.gz, a yfinance option-chain snapshot taken
2026-11-19 15:03 UTC (10:03 New York), committed so the fit reproduces exactly.
Discounting: flat 3.9%, the 30-day average SOFR (3.916% on 2026-11-12, FRED SOFR30DAYAVG).
Forwards come from put-call parity per expiry; see marketdata/option_chain.py.

1. Surface: OTM quotes, 14 days to 2.2 years, standardized moneyness in [-3, 2].
2. Split: every 4th strike of each expiry is held out; the fit sees the other 75%.
3. Fit: vega-weighted least squares from 4 starts; report whether they agree.
4. Errors in implied-vol points, in-sample and held out, against two baselines fitted on the
   same in-sample quotes: one flat vol for everything, and a flat vol per expiry (term
   structure but no smile).
5. Gate: parameter recovery on SPY's own strike grid. Quotes regenerated from the fitted
   parameters by the independent adaptive-quadrature pricer must calibrate back to them.
   This proves the reported parameters are identified by this surface, not just one of many
   fits.

    uv run python -m mc_engine.experiments.calibration
    uv run python -m mc_engine.experiments.calibration --no-figure
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from marketdata.option_chain import build_surface, implied_forwards, load_yfinance_snapshot
from mc_engine import bs_vega
from mc_engine.heston import HestonParams, heston_price
from mc_engine.heston_calibration import calibrate, call_prices_fwd, model_implied_vols
from mc_engine.implied_vol import implied_vol

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data" / "snapshots" / "spy_options_2026-11-19.csv.gz"
RATE = 0.039
FIGURE = ROOT / "docs" / "figures" / "spy_heston_smiles.png"


def rmse(x: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def split(surface: pd.DataFrame) -> pd.Series:
    """True for held-out quotes: every 4th strike within each expiry."""
    order = surface.groupby("expiry").cumcount()
    return order % 4 == 3


def recovery_gate(surface: pd.DataFrame, true: HestonParams) -> bool:
    """Regenerate quotes from ``true`` on every 6th SPY strike with the adaptive-quadrature
    pricer, calibrate, and require the parameters back."""
    sub = surface.iloc[::6].copy()
    mids, ivs, vegas = [], [], []
    for row in sub.itertuples():
        r = -np.log(row.df) / row.t
        call = row.df * heston_price(row.forward, row.strike, row.t, 0.0, true)
        mid = call if row.kind == "call" else call - row.df * (row.forward - row.strike)
        iv = implied_vol(mid, row.forward, row.strike, row.t, r, r, row.kind)
        mids.append(mid)
        ivs.append(iv)
        vegas.append(float(bs_vega(row.forward, row.strike, row.t, r, iv, r)))
    sub["mid"], sub["iv"], sub["vega"] = mids, ivs, vegas
    got = calibrate(sub).params
    print(f"\n## Gate: parameter recovery on SPY's strike grid ({len(sub)} regenerated quotes)\n")
    print("| Parameter | True (= SPY fit) | Recovered |")
    print("|---|---:|---:|")
    ok = True
    for name, tol, rel in (
        ("kappa", 0.03, True),
        ("theta", 0.01, True),
        ("xi", 0.01, True),
        ("rho", 0.005, False),
        ("v0", 0.01, True),
    ):
        a, b = getattr(true, name), getattr(got, name)
        ok &= abs(b - a) <= (tol * abs(a) if rel else tol)
        print(f"| {name} | {a:.4f} | {b:.4f} |")
    return ok


def plot(surface: pd.DataFrame, held_out: pd.Series, p: HestonParams) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    surface_c, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    ts = surface.groupby("expiry")["t"].first().sort_values()
    picks = [
        ts.index[np.argmin(np.abs(ts.to_numpy() - x))] for x in (0.04, 0.12, 0.33, 0.6, 1.1, 2.2)
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), dpi=150, sharey=False)
    fig.patch.set_facecolor(surface_c)
    for ax, expiry in zip(axes.flat, picks, strict=True):
        g = surface[surface.expiry == expiry]
        h = held_out[g.index]
        t, fwd, disc = g.t.iloc[0], g.forward.iloc[0], g.df.iloc[0]
        r = -np.log(disc) / t
        lm = np.linspace(g.log_moneyness.min(), g.log_moneyness.max(), 120)
        kf = np.exp(lm)
        calls = call_prices_fwd(kf, t, p) * fwd * disc
        model_iv = [
            implied_vol(
                c if k >= 1 else c - disc * fwd * (1 - k),
                fwd,
                k * fwd,
                t,
                r,
                r,
                "call" if k >= 1 else "put",
            )
            for c, k in zip(calls, kf, strict=True)
        ]
        ax.set_facecolor(surface_c)
        ax.scatter(
            g.log_moneyness[~h],
            100 * g.iv[~h],
            s=7,
            color=blue,
            alpha=0.55,
            lw=0,
            label="Market (fitted)",
        )
        ax.scatter(
            g.log_moneyness[h],
            100 * g.iv[h],
            s=12,
            facecolors="none",
            edgecolors=blue,
            lw=0.8,
            label="Market (held out)",
        )
        ax.plot(lm, 100 * np.array(model_iv), color=orange, lw=2, label="Heston fit")
        ax.set_title(f"{expiry}  (T = {t:.2f}y)", loc="left", color=ink, fontsize=9.5)
        ax.grid(True, color=grid, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    for ax in axes[1]:
        ax.set_xlabel("ln(K / F)", color=ink2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied vol (%)", color=ink2, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=ink,
        bbox_to_anchor=(0.99, 0.955),
    )
    fig.suptitle(
        "One Heston parameter set across 22 SPY expiries: close mid-curve, misses the "
        "short-dated skew",
        x=0.01,
        y=0.995,
        ha="left",
        color=ink,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.935,
        "SPY option snapshot 2026-11-19 10:03 New York · OTM mid quotes · "
        "every 4th strike held out of the fit",
        color=ink2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, facecolor=surface_c)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="The occurrence of roundoff error")

    raw = load_yfinance_snapshot(SNAPSHOT)
    surface = build_surface(raw, RATE)
    held_out = split(surface)
    fwd = implied_forwards(raw, RATE)
    print(
        f"## SPY snapshot {raw.snapshot_utc.iloc[0]}: spot {raw.spot.iloc[0]:.2f}, "
        f"{len(raw):,} raw quotes → {len(surface):,} usable OTM quotes across "
        f"{surface.expiry.nunique()} expiries ({surface.t.min():.3f}–{surface.t.max():.2f}y); "
        f"{(~held_out).sum():,} fitted, {held_out.sum():,} held out."
    )
    print(
        f"Forwards from put-call parity agree across near-ATM strikes to within "
        f"${fwd.forward_range.median():.2f} (median range).\n"
    )

    fit_surface = surface[~held_out]
    res = calibrate(fit_surface)
    p = res.params
    print("## Fitted parameters\n")
    print(
        f"kappa = {p.kappa:.3f}, theta = {p.theta:.4f} (long-run vol {np.sqrt(p.theta):.1%}), "
        f"xi = {p.xi:.3f}, rho = {p.rho:.3f}, v0 = {p.v0:.4f} (spot vol {np.sqrt(p.v0):.1%}). "
        f"Feller {'satisfied' if p.feller else 'violated'} "
        f"(2·kappa·theta = {2 * p.kappa * p.theta:.3f} vs xi² = {p.xi**2:.3f}).\n"
    )
    print("| Start (kappa, theta, xi, rho, v0) | Converged to | Cost |")
    print("|---|---|---:|")
    for x0, x, cost in res.runs:
        print(
            f"| {', '.join(f'{v:g}' for v in x0)} | {', '.join(f'{v:.4f}' for v in x)} "
            f"| {cost:.4f} |"
        )

    model_iv = model_implied_vols(surface, p)
    err = 100 * (model_iv - surface.iv)
    flat = 100 * (surface.iv[~held_out].mean() - surface.iv)
    per_exp_mean = surface[~held_out].groupby("expiry").iv.mean()
    per_exp = 100 * (surface.expiry.map(per_exp_mean) - surface.iv)
    print("\n## Fit error, implied-vol points (1.00 = 1% vol)\n")
    print("| Model | In-sample RMSE | Held-out RMSE |")
    print("|---|---:|---:|")
    for label, e in (
        ("One flat vol (Black–Scholes)", flat),
        ("Flat vol per expiry (term structure, no smile)", per_exp),
        ("Heston, 5 parameters", err),
    ):
        print(f"| {label} | {rmse(e[~held_out]):.3f} | {rmse(e[held_out]):.3f} |")

    by_exp = pd.DataFrame(
        {"t": surface.t, "expiry": surface.expiry, "err": err, "held": held_out}
    ).groupby("expiry")
    print("\n| Expiry | T (years) | Quotes | In-sample RMSE | Held-out RMSE | Mean error |")
    print("|---|---:|---:|---:|---:|---:|")
    for expiry, g in sorted(by_exp, key=lambda kv: kv[1].t.iloc[0]):
        print(
            f"| {expiry} | {g.t.iloc[0]:.3f} | {len(g)} | {rmse(g.err[~g.held]):.2f} "
            f"| {rmse(g.err[g.held]):.2f} | {g.err.mean():+.2f} |"
        )

    ok = recovery_gate(surface, p)
    if not args.no_figure:
        plot(surface, held_out, p)
        print(f"\nFigure written to {FIGURE}")
    print(f"\nStep 6 gate (parameter recovery): {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
