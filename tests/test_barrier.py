"""Down-and-out call: closed form sanity, the naive estimator's bias, and the bridge fix."""

import numpy as np
import pytest

from mc_engine import bs_price
from mc_engine.barrier import down_and_out_call_price, price_down_and_out_call

S0, K, B, R, Q, SIGMA, T = 100.0, 100.0, 90.0, 0.05, 0.01, 0.25, 1.0
Z_TOL = 4.0
N = 1 << 17


def test_closed_form_limits():
    vanilla = float(bs_price(S0, K, T, R, SIGMA, Q, "call"))
    # A barrier far below spot almost never triggers.
    assert down_and_out_call_price(S0, K, 1e-3, T, R, SIGMA, Q) == pytest.approx(vanilla, rel=1e-9)
    # Starting just above the barrier, the option is almost surely knocked out.
    assert down_and_out_call_price(S0, K, S0 * (1 - 1e-6), T, R, SIGMA, Q) < 1e-3
    # Knock-out can only remove value, and a higher barrier removes more.
    prices = [down_and_out_call_price(S0, K, b, T, R, SIGMA, Q) for b in (60, 75, 90, 95)]
    assert prices[0] <= vanilla and np.all(np.diff(prices) < 0)


@pytest.mark.parametrize("n_steps", [4, 16, 64])
def test_bridge_correction_is_unbiased_at_any_step_count(n_steps):
    ref = down_and_out_call_price(S0, K, B, T, R, SIGMA, Q)
    res = price_down_and_out_call(S0, K, B, T, R, SIGMA, Q, n_steps, N, seed=n_steps)
    assert abs(res.z_score(ref)) < Z_TOL, (n_steps, res, ref)


def test_naive_monitoring_overprices_and_converges_slowly():
    ref = down_and_out_call_price(S0, K, B, T, R, SIGMA, Q)
    bias = []
    for n_steps in (16, 64, 256):
        res = price_down_and_out_call(
            S0, K, B, T, R, SIGMA, Q, n_steps, N, seed=n_steps, bridge_correction=False
        )
        assert res.z_score(ref) > 5, (n_steps, res, ref)  # missed knock-outs: too expensive
        bias.append(res.price - ref)
    # Bias ~ 1/sqrt(n): quadrupling the steps should roughly halve it.
    assert 0.35 < bias[2] / bias[1] < 0.7
    assert 0.35 < bias[1] / bias[0] < 0.7


def test_bridge_correction_also_lowers_variance():
    naive = price_down_and_out_call(
        S0, K, B, T, R, SIGMA, Q, 64, N, seed=9, bridge_correction=False
    )
    bridge = price_down_and_out_call(S0, K, B, T, R, SIGMA, Q, 64, N, seed=9)
    assert bridge.std_error < naive.std_error


def test_sobol_bridge_corrected():
    ref = down_and_out_call_price(S0, K, B, T, R, SIGMA, Q)
    res = price_down_and_out_call(S0, K, B, T, R, SIGMA, Q, 16, N, seed=3, sampler="sobol")
    assert abs(res.z_score(ref)) < Z_TOL


@pytest.mark.parametrize("b", [0.0, 100.0, 120.0])
def test_rejects_unsupported_barriers(b):
    with pytest.raises(ValueError):
        down_and_out_call_price(S0, 95.0, b, T, R, SIGMA, Q)
