"""Error vs path count on log-log axes: plain Monte Carlo should fall with slope -1/2.

For each path count N we run independent replications, measure the RMSE of the estimate
against the Black–Scholes price, and compare it with the standard error the estimator reports.
The two lines should overlap; if they don't, the reported error bars can't be trusted.

    uv run python -m mc_engine.experiments.convergence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mc_engine import bs_price, price_european

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.20
PATH_COUNTS = [1 << p for p in range(8, 21)]  # 256 ... 1,048,576
REPLICATIONS = 64
SEED = 20261118
FIGURE = Path(__file__).resolve().parents[2] / "docs" / "figures" / "mc_convergence.png"


def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    ref = float(bs_price(S0, K, T, R, SIGMA))
    root = np.random.SeedSequence(SEED)
    rmse, mean_se = [], []
    for n, seq in zip(PATH_COUNTS, root.spawn(len(PATH_COUNTS)), strict=True):
        results = [
            price_european(S0, K, T, R, SIGMA, n_paths=n, seed=s) for s in seq.spawn(REPLICATIONS)
        ]
        err = np.array([r.price - ref for r in results])
        rmse.append(np.sqrt(np.mean(err**2)))
        mean_se.append(np.mean([r.std_error for r in results]))
    n = np.array(PATH_COUNTS, dtype=float)
    rmse, mean_se = np.array(rmse), np.array(mean_se)
    slope = float(np.polyfit(np.log(n), np.log(rmse), 1)[0])
    return n, rmse, mean_se, slope


def plot(n: np.ndarray, rmse: np.ndarray, mean_se: np.ndarray, slope: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    # Reference line anchored at the first point, so only its slope carries meaning.
    ax.plot(
        n,
        rmse[0] * np.sqrt(n[0] / n),
        color=ink2,
        lw=1.2,
        ls=(0, (4, 3)),
        zorder=1,
        label="Reference slope −½",
    )
    ax.plot(
        n,
        rmse,
        color=blue,
        lw=2,
        marker="o",
        ms=5.5,
        mec=surface,
        mew=1.5,
        zorder=3,
        label=f"Empirical RMSE vs Black–Scholes ({REPLICATIONS} runs per N)",
    )
    ax.plot(
        n,
        mean_se,
        color=orange,
        lw=2,
        marker="s",
        ms=5,
        mec=surface,
        mew=1.5,
        zorder=2,
        label="Mean reported standard error",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Paths (N)", color=ink2)
    ax.set_ylabel("Pricing error ($)", color=ink2)
    ax.set_title(
        f"Monte Carlo error falls as 1/√N  (fitted slope {slope:.3f})".replace("-", "−"),
        loc="left",
        color=ink,
        fontsize=12,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0,
        1.035,
        f"ATM European call, S = K = {S0:.0f}, T = {T:.0f}y, r = {R:.0%}, "
        f"σ = {SIGMA:.0%} · fixed seed {SEED}",
        transform=ax.transAxes,
        color=ink2,
        fontsize=8.5,
    )

    ax.grid(True, which="major", color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ink, loc="upper right")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE, facecolor=surface)
    plt.close(fig)


def main() -> None:
    n, rmse, mean_se, slope = run()
    print("| Paths | RMSE vs BS | Mean reported SE | RMSE / SE |")
    print("|---:|---:|---:|---:|")
    for ni, e, s in zip(n, rmse, mean_se, strict=True):
        print(f"| {int(ni):,} | {e:.5f} | {s:.5f} | {e / s:.2f} |")
    print(f"\nFitted log-log slope: {slope:.3f} (theory: -0.5)")
    plot(n, rmse, mean_se, slope)
    print(f"Figure written to {FIGURE}")


if __name__ == "__main__":
    main()
