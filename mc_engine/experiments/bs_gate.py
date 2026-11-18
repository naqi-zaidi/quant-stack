"""Phase 1 gate: plain Monte Carlo against Black–Scholes.

Part 1 prices a 12-contract grid at 1M paths each, one independent seed per contract.
Part 2 is the actual gate. "Within one standard error" is a coverage claim, so it is tested
as one: if the estimator is unbiased and its standard error is right, z = (MC - BS) / SE is
N(0, 1), so about 68.3% of runs land inside |z| < 1 and 95.4% inside |z| < 2. Twelve z-scores
can't check that (4/12 inside |z| < 1 happens), so part 2 reruns every contract with 100
independent seeds and checks coverage and the spread of z over all 1,200 runs. Coverage far
*above* 68% would fail too: it would mean the error bars are overstated.

    uv run python -m mc_engine.experiments.bs_gate
"""

from __future__ import annotations

import numpy as np

from mc_engine import bs_price, price_european

S0, R, Q, SIGMA = 100.0, 0.05, 0.01, 0.20
N_PATHS = 1_000_000
SEED = 20261118
COVERAGE_SEEDS = 100  # independent runs per contract in part 2
COVERAGE_PATHS = 50_000

CONTRACTS = [
    (kind, k, t) for kind in ("call", "put") for t in (0.5, 1.0) for k in (90.0, 100.0, 110.0)
]


def run() -> list[dict]:
    seeds = np.random.SeedSequence(SEED).spawn(len(CONTRACTS))
    rows = []
    for (kind, k, t), seed in zip(CONTRACTS, seeds, strict=True):
        ref = float(bs_price(S0, k, t, R, SIGMA, Q, kind))
        res = price_european(S0, k, t, R, SIGMA, Q, kind, n_paths=N_PATHS, seed=seed)
        rows.append(
            {
                "kind": kind,
                "k": k,
                "t": t,
                "bs": ref,
                "mc": res.price,
                "se": res.std_error,
                "z": res.z_score(ref),
            }
        )
    return rows


def coverage() -> np.ndarray:
    """z-scores from COVERAGE_SEEDS independent runs of every contract."""
    root = np.random.SeedSequence([SEED, 1])
    z = []
    for (kind, k, t), seq in zip(CONTRACTS, root.spawn(len(CONTRACTS)), strict=True):
        ref = float(bs_price(S0, k, t, R, SIGMA, Q, kind))
        for s in seq.spawn(COVERAGE_SEEDS):
            res = price_european(S0, k, t, R, SIGMA, Q, kind, n_paths=COVERAGE_PATHS, seed=s)
            z.append(res.z_score(ref))
    return np.array(z)


def main() -> None:
    rows = run()
    print(f"S0={S0}, r={R}, q={Q}, sigma={SIGMA}, {N_PATHS:,} paths per contract, seed {SEED}\n")
    print("| Option | K | T | Black–Scholes | Monte Carlo | Std error | z |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['kind']} | {r['k']:.0f} | {r['t']} | {r['bs']:.4f} | {r['mc']:.4f} "
            f"| {r['se']:.4f} | {r['z']:+.2f} |"
        )
    z = coverage()
    n = z.size
    print(f"\nCoverage over {n:,} independent runs ({COVERAGE_PATHS:,} paths each):\n")
    print("| Check | Observed | Expected if SE is right |")
    print("|---|---:|---:|")
    passed = True
    for width, p in ((1, 0.6827), (2, 0.9545)):
        obs = np.mean(np.abs(z) < width)
        band = 3 * np.sqrt(p * (1 - p) / n)  # 3-sigma binomial band
        passed &= abs(obs - p) <= band
        print(f"| share inside \\|z\\| < {width} | {obs:.1%} | {p:.1%} ± {band:.1%} |")
    mean_band, sd_band = 3 / np.sqrt(n), 3 / np.sqrt(2 * (n - 1))
    passed &= abs(z.mean()) <= mean_band and abs(z.std(ddof=1) - 1) <= sd_band
    print(f"| mean z | {z.mean():+.3f} | 0 ± {mean_band:.3f} |")
    print(f"| sd of z | {z.std(ddof=1):.3f} | 1 ± {sd_band:.3f} |")
    print(f"\nGate: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
