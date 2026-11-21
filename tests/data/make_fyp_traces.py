"""Records the FYP's own policy, regime gate and anomaly scorer on random inputs, as the golden
fixture fyp_traces.json that tests/test_risk.py checks mm_strategy.risk against.

Needs a checkout of https://github.com/naqi-zaidi/fyp-trading-risk-guardrails (its v10 package)
on PYTHONPATH, with numpy, scikit-learn and pyyaml. From the repository root:

    PYTHONPATH=/path/to/fyp-trading-risk-guardrails uv run --no-project --with numpy \
        --with scikit-learn --with pyyaml python tests/data/make_fyp_traces.py <fyp commit>

Parameters are the FYP's frozen config.yaml values. The hedge rule is kept out by a zero slope
(it isn't ported). Labels are stored as indices into REGIMES, LEVELS and (allow, throttle,
pause); inputs are rounded to 3 decimals before the FYP sees them.
"""

import json
import sys
from pathlib import Path

import numpy as np
from v10.detectors.anomaly import AnomalyDetector
from v10.detectors.regime import RegimeDetector
from v10.policy.rules import DecisionPolicy

REGIMES = ["calm", "trend_up", "trend_down", "volatile", "stressed"]
LEVELS = ["none", "moderate", "severe"]
rng = np.random.default_rng(42)
N = 3000

# Policy: regimes and anomalies in runs, so transitions, timers and cooldowns all get exercised.
dp = DecisionPolicy(throttle_size_mult=0.33, throttle_duration=5, pause_duration=10,
                    hedge_cooldown=10, persistence_label=2, persistence_moderate=1,
                    persistence_severe=1)
pol_in, pol_out = [], []
r, a = "calm", "none"
for _ in range(N):
    if rng.random() < 0.3:
        r = REGIMES[rng.choice(5, p=[0.4, 0.15, 0.15, 0.15, 0.15])]
    if rng.random() < 0.4:
        a = LEVELS[rng.choice(3, p=[0.6, 0.25, 0.15])]
    d = dp.decide(regime_label=r, regime_confidence=1.0, anomaly_level=a, anomaly_score=0.0,
                  if_confirmed=False, trend_strength=0.0, event_window_high=False,
                  event_window_med=False, z_ret=0.0, z_range=0.0, atr_14=1.0,
                  current_price=100.0, slope_short=0.0)
    pol_in.append([REGIMES.index(r), LEVELS.index(a)])
    pol_out.append(["allow", "throttle", "pause"].index(d["action"]))

# Regime gate: provisional labels, returns and sigmas through CUSUM and persistence.
rd = RegimeDetector(k=5, persistence_bars=2, cusum_k_mult=1.5, cusum_h_mult=3.0)
reg_in, reg_out = [], []
prov = "calm"
for _ in range(N):
    if rng.random() < 0.35:
        prov = REGIMES[rng.integers(5)]
    ret = round(float(rng.normal(0, 1) * (3 if prov in ("volatile", "stressed") else 1)), 3)
    sig = round(float(rng.uniform(0.5, 1.5)), 3)
    reg_in.append([REGIMES.index(prov), ret, sig])
    reg_out.append(REGIMES.index(rd.confirm_transition(ret, sig, prov)))

# Anomaly levels from the two z-scores.
ad = AnomalyDetector(z_moderate=2.0, z_severe=3.0, persistence_moderate=1, persistence_severe=1)
an_in, an_out = [], []
for _ in range(N):
    zr, zg = (round(float(x), 3) for x in rng.normal(0, 1.6, 2))
    reg = int(rng.integers(5))
    an_in.append([zr, zg, reg])
    an_out.append(LEVELS.index(ad.score_z(zr, zg, REGIMES[reg]).anomaly_level))

out = {"fyp_commit": sys.argv[1] if len(sys.argv) > 1 else "",
       "policy": {"inputs": pol_in, "actions": pol_out},
       "regime": {"inputs": reg_in, "labels": reg_out},
       "anomaly": {"inputs": an_in, "levels": an_out}}
Path(__file__).with_name("fyp_traces.json").write_text(json.dumps(out, separators=(",", ":")))
print({k: (len(v["inputs"]) if isinstance(v, dict) else v) for k, v in out.items()})
