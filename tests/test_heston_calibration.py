"""Fast Heston pricer and calibrator.

The pricer is checked against the adaptive-quadrature pricer (itself benchmarked against
published values). The calibrator is checked by parameter recovery: quotes generated from
known parameters with the *independent* adaptive-quadrature pricer must give those parameters
back.
"""

import numpy as np
import pandas as pd
import pytest

from mc_engine import bs_vega
from mc_engine.heston import HestonParams, heston_price
from mc_engine.heston_calibration import calibrate, call_prices_fwd, model_implied_vols
from mc_engine.implied_vol import implied_vol

CASES = [
    HestonParams(2.0, 0.04, 0.5, -0.7, 0.04),
    HestonParams(0.5, 0.04, 1.0, -0.9, 0.04),  # Feller badly violated: heavy CF tail
    HestonParams(1.0, 0.09, 1.5, -0.8, 0.06),
    HestonParams(0.2, 0.08, 2.0, -0.95, 0.02),
]


@pytest.mark.parametrize("p", CASES)
@pytest.mark.parametrize("t", [0.04, 0.5, 2.2])
def test_fast_pricer_matches_adaptive_quadrature(p, t):
    kf = np.exp(np.linspace(-3, 2, 11) * 0.2 * np.sqrt(t))
    ref = np.array([heston_price(1.0, k, t, 0.0, p) for k in kf])
    np.testing.assert_allclose(call_prices_fwd(kf, t, p), ref, atol=1e-9)


def _synthetic_surface(true: HestonParams, rate=0.04) -> pd.DataFrame:
    rows = []
    for t in (0.1, 0.3, 0.75, 1.5):
        fwd, disc = 500.0 * np.exp(0.03 * t), np.exp(-rate * t)
        for m in np.linspace(-2.5, 1.5, 9):
            k = fwd * np.exp(m * 0.2 * np.sqrt(t))
            kind = "put" if k < fwd else "call"
            call = disc * heston_price(fwd, k, t, 0.0, true)
            mid = call if kind == "call" else call - disc * (fwd - k)
            iv = implied_vol(mid, fwd, k, t, rate, rate, kind)
            rows.append(
                {
                    "t": t,
                    "kind": kind,
                    "strike": k,
                    "mid": mid,
                    "forward": fwd,
                    "df": disc,
                    "iv": iv,
                    "vega": float(bs_vega(fwd, k, t, rate, iv, rate)),
                }
            )
    return pd.DataFrame(rows)


def test_calibration_recovers_known_parameters():
    true = HestonParams(3.5, 0.055, 1.3, -0.73, 0.043)
    surf = _synthetic_surface(true)
    res = calibrate(surf)
    got = res.params
    assert got.kappa == pytest.approx(true.kappa, rel=0.02)
    assert got.theta == pytest.approx(true.theta, rel=0.01)
    assert got.xi == pytest.approx(true.xi, rel=0.01)
    assert got.rho == pytest.approx(true.rho, abs=0.005)
    assert got.v0 == pytest.approx(true.v0, rel=0.01)
    assert np.sqrt(np.mean(res.iv_errors(surf) ** 2)) < 0.01  # vol points


def test_model_implied_vols_reproduce_the_surface_at_the_true_parameters():
    true = HestonParams(2.0, 0.04, 0.6, -0.6, 0.05)
    surf = _synthetic_surface(true)
    np.testing.assert_allclose(model_implied_vols(surf, true), surf["iv"], atol=1e-7)


def test_every_start_is_reported():
    surf = _synthetic_surface(HestonParams(2.0, 0.04, 0.6, -0.6, 0.05))
    starts = ((2.0, 0.04, 0.5, -0.7, 0.04), (1.0, 0.06, 1.0, -0.8, 0.03))
    res = calibrate(surf, starts=starts)
    assert len(res.runs) == 2
    assert res.cost == min(cost for _, _, cost in res.runs)
