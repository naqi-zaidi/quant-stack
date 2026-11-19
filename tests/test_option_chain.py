"""Option-chain loader on a synthetic chain generated from known inputs, so every recovered
quantity has an exact answer."""

import numpy as np
import pandas as pd
import pytest

from marketdata.option_chain import build_surface, implied_forwards, load_yfinance_snapshot
from mc_engine import bs_price

SPOT, RATE, DIV = 500.0, 0.04, 0.012
SNAP = "2026-11-19T15:00:00+00:00"


def _smile(k, f, t):
    return 0.20 - 0.10 * np.log(k / f) + 0.3 * np.log(k / f) ** 2  # downward skew


def _write_chain(path, expiries=("2026-12-18", "2027-06-17"), half_spread=0.02):
    snap = pd.Timestamp(SNAP)
    rows = []
    for exp in expiries:
        close = pd.Timestamp(exp).tz_localize("America/New_York") + pd.Timedelta(hours=16)
        t = (close.tz_convert("UTC") - snap).total_seconds() / (365 * 86400)
        f = SPOT * np.exp((RATE - DIV) * t)
        for k in np.arange(300.0, 701.0, 5.0):
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, k, t, RATE, _smile(k, f, t), DIV, kind))
                rows.append(
                    {
                        "strike": k,
                        "bid": px - half_spread,
                        "ask": px + half_spread,
                        "kind": kind,
                        "expiry": exp,
                        "spot": SPOT,
                        "snapshot_utc": SNAP,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def chain(tmp_path):
    p = tmp_path / "chain.csv"
    _write_chain(p)
    return load_yfinance_snapshot(p)


def test_time_to_expiry_counts_to_the_new_york_close(chain):
    t = chain.groupby("expiry")["t"].first()
    # 19 Nov 15:00 UTC to 18 Dec 21:00 UTC (16:00 EST) = 29 days + 6 hours.
    assert t["2026-12-18"] == pytest.approx((29 + 6 / 24) / 365, abs=1e-9)


def test_forwards_recovered_from_parity(chain):
    fw = implied_forwards(chain, RATE).set_index("expiry")
    for exp, row in fw.iterrows():
        assert row.forward == pytest.approx(SPOT * np.exp((RATE - DIV) * row.t), rel=1e-9), exp
        assert row.n_pairs >= 3


def test_surface_recovers_the_generating_smile(chain):
    surf = build_surface(chain, RATE, min_t=0.0)
    # Quotes are symmetric around the true price, so mids are exact and vols match the smile.
    expected = _smile(surf.strike, surf.forward, surf.t)
    np.testing.assert_allclose(surf.iv, expected, atol=1e-8)


def test_surface_keeps_only_out_of_the_money_quotes(chain):
    surf = build_surface(chain, RATE, min_t=0.0)
    assert (surf.loc[surf.kind == "put", "strike"] < surf.loc[surf.kind == "put", "forward"]).all()
    assert (
        surf.loc[surf.kind == "call", "strike"] >= surf.loc[surf.kind == "call", "forward"]
    ).all()


def test_filters_apply(chain):
    wide = build_surface(chain, RATE, min_t=0.0, moneyness=(-1.0, 1.0))
    full = build_surface(chain, RATE, min_t=0.0, moneyness=(-10.0, 10.0))
    assert len(wide) < len(full)
    assert wide.m.between(-1.0, 1.0).all()
    assert build_surface(chain, RATE, min_t=0.1).expiry.nunique() == 1  # drops the 1-month
    assert (build_surface(chain, RATE, min_t=0.0, min_mid=1.0).mid >= 1.0).all()


def test_rejects_snapshots_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"strike": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_yfinance_snapshot(p)
