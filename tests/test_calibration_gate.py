"""The calibration gate must pass honest error bars and fail wrong ones.

Synthetic z-scores stand in for estimator runs: if a method reports SE but its true error is
c * SE, its z-scores are N(0, c^2). Using the gate's own sample sizes (4 contracts x 200 runs)
shows its bands are wide enough to avoid false alarms and narrow enough to catch real errors.
"""

import numpy as np
import pytest

from mc_engine.experiments.variance_reduction import CONTRACTS, RUNS, calibration_ok

N_Z = len(CONTRACTS) * RUNS


def _synthetic(scale: float, seed: int):
    rng = np.random.default_rng(seed)
    z = scale * rng.standard_normal(N_Z)
    # Observed variance over RUNS runs vs the (wrong) reported SE^2, one ratio per contract.
    obs = scale**2 * rng.chisquare(RUNS - 1, size=len(CONTRACTS)) / (RUNS - 1)
    return z, 1.0 / obs


def test_honest_error_bars_pass_almost_always():
    passes = [calibration_ok(*_synthetic(1.0, s), RUNS) for s in range(500)]
    assert np.mean(passes) > 0.99


@pytest.mark.parametrize(
    "scale",
    [1.25, 0.8],  # error bars 20% too small, or 25% too large
)
def test_wrong_error_bars_fail(scale):
    fails = [not calibration_ok(*_synthetic(scale, s), RUNS) for s in range(200)]
    assert np.mean(fails) > 0.99
