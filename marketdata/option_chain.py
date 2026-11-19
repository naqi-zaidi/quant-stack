"""Option-chain snapshots (as saved from yfinance) to a clean implied-volatility surface.

Steps, and why each one is there:

1. Time to expiry. Equity options stop trading at 16:00 New York time on expiry day. T is
   measured from the snapshot timestamp in years (ACT/365), so a quote taken mid-morning has
   the right fraction of a day left.
2. Forward per expiry, from put-call parity at the money: C - P = DF (F - K), so
   F = K + (C - P) / DF for each strike within 2% of spot, then the median across strikes.
   The discount factor comes from a supplied short rate, not from the parity slope. SPY
   options are American, and the early-exercise premium on puts grows with strike, which tilts
   the slope; a $0.50 quote spread also swamps it at short expiries. The forward, pinned at the
   money, is robust to both. DF itself barely moves implied vols (an error of 0.01 shifts them
   by about 0.04 vol points).
3. Out-of-the-money quotes only: puts below the forward, calls above. ITM American options
   carry early-exercise premium and wide spreads; OTM ones are the liquid, near-European side.
4. Quote filters: positive bid, ask above bid, mid at least ``min_mid`` and a relative spread
   of at most ``max_spread``.
5. Implied vol by Black-76 on the forward, and standardized moneyness
   m = ln(K / F) / (sigma_atm sqrt(T)), so a strike band means the same thing at every
   maturity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mc_engine.black_scholes import bs_vega
from mc_engine.implied_vol import implied_vol

SECONDS_PER_YEAR = 365.0 * 86400.0


def load_yfinance_snapshot(path: str | Path) -> pd.DataFrame:
    """Raw snapshot plus ``t`` (years to the 16:00 New York close on expiry) and ``mid``."""
    df = pd.read_csv(path)
    required = {"strike", "bid", "ask", "kind", "expiry", "spot", "snapshot_utc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"snapshot is missing columns: {sorted(missing)}")
    snap = pd.Timestamp(df["snapshot_utc"].iloc[0])
    close = (
        pd.to_datetime(df["expiry"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    ).dt.tz_convert("UTC")
    df["t"] = (close - snap).dt.total_seconds() / SECONDS_PER_YEAR
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    return df[df["t"] > 0].reset_index(drop=True)


def implied_forwards(df: pd.DataFrame, rate: float, band: float = 0.02) -> pd.DataFrame:
    """Per-expiry forward from put-call parity on strikes within ``band`` of spot.

    Returns expiry, t, forward, the number of call/put pairs used, and the range of the
    per-strike forward estimates (a direct read on quote noise and early-exercise effects).
    """
    spot = float(df["spot"].iloc[0])
    valid = df[(df["bid"] > 0) & (df["ask"] > df["bid"])]
    rows = []
    for expiry, g in valid.groupby("expiry"):
        t = float(g["t"].iloc[0])
        calls = g[g["kind"] == "call"].set_index("strike")["mid"]
        puts = g[g["kind"] == "put"].set_index("strike")["mid"]
        strikes = calls.index.intersection(puts.index)
        strikes = strikes[np.abs(strikes / spot - 1.0) < band]
        if len(strikes) < 3:
            continue
        per_strike = strikes.to_numpy() + (calls[strikes] - puts[strikes]).to_numpy() / np.exp(
            -rate * t
        )
        rows.append(
            {
                "expiry": expiry,
                "t": t,
                "forward": float(np.median(per_strike)),
                "n_pairs": len(strikes),
                "forward_range": float(np.ptp(per_strike)),
            }
        )
    return pd.DataFrame(rows)


def build_surface(
    df: pd.DataFrame,
    rate: float,
    *,
    min_t: float = 14 / 365,
    min_mid: float = 0.10,
    max_spread: float = 0.5,
    moneyness: tuple[float, float] = (-3.0, 2.0),
) -> pd.DataFrame:
    """Clean OTM implied-vol surface: one row per usable quote.

    Columns: expiry, t, kind, strike, bid, ask, mid, forward, df (discount factor), iv,
    log_moneyness, m (standardized moneyness), vega (Black-76, per 1.00 of vol).
    """
    forwards = implied_forwards(df, rate).set_index("expiry")
    out = []
    for expiry, g in df.groupby("expiry"):
        if expiry not in forwards.index:
            continue
        t, fwd = float(forwards.at[expiry, "t"]), float(forwards.at[expiry, "forward"])
        if t < min_t:
            continue
        otm = g[
            ((g["kind"] == "put") & (g["strike"] < fwd))
            | ((g["kind"] == "call") & (g["strike"] >= fwd))
        ].copy()
        spread = (otm["ask"] - otm["bid"]) / otm["mid"]
        otm = otm[
            (otm["bid"] > 0)
            & (otm["ask"] > otm["bid"])
            & (otm["mid"] >= min_mid)
            & (spread <= max_spread)
        ]
        if otm.empty:
            continue
        # Black-76 on the forward: bs_price with s = F and q = r discounts F by e^{-rT}.
        otm["iv"] = [
            implied_vol(m, fwd, k, t, rate, rate, kind)
            for m, k, kind in zip(otm["mid"], otm["strike"], otm["kind"], strict=True)
        ]
        otm = otm[np.isfinite(otm["iv"])]
        otm["forward"], otm["df"] = fwd, np.exp(-rate * t)
        otm["log_moneyness"] = np.log(otm["strike"] / fwd)
        atm_iv = float(otm.loc[otm["log_moneyness"].abs().idxmin(), "iv"])
        otm["m"] = otm["log_moneyness"] / (atm_iv * np.sqrt(t))
        otm = otm[(otm["m"] >= moneyness[0]) & (otm["m"] <= moneyness[1])]
        otm["vega"] = bs_vega(fwd, otm["strike"].to_numpy(), t, rate, otm["iv"].to_numpy(), rate)
        out.append(otm)
    cols = [
        "expiry",
        "t",
        "kind",
        "strike",
        "bid",
        "ask",
        "mid",
        "forward",
        "df",
        "iv",
        "log_moneyness",
        "m",
        "vega",
    ]
    return (
        pd.concat(out, ignore_index=True)[cols].sort_values(["t", "strike"]).reset_index(drop=True)
    )
