"""Save an option-chain snapshot from Yahoo Finance in the format ``option_chain`` reads.

    python -m pip install --user yfinance
    python -m marketdata.yfinance_snapshot SPY            # writes spy_options_<date>.csv

Run it during US market hours: before the open, bid/ask fields come back empty or stale.
yfinance is an unofficial scraper, so snapshots are saved to disk and committed rather than
re-downloaded; data/snapshots/ holds the one the step 6 calibration uses.
"""

from __future__ import annotations

import argparse
import datetime as dt


def snapshot(ticker: str, min_days: int = 7, max_days: int = 800) -> str:
    import pandas as pd
    import yfinance as yf  # optional dependency, imported only when snapshotting

    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    today = dt.date.today()
    frames = []
    for expiry in tk.options:
        days = (dt.date.fromisoformat(expiry) - today).days
        if min_days <= days <= max_days:
            chain = tk.option_chain(expiry)
            for kind, df in (("call", chain.calls), ("put", chain.puts)):
                frames.append(df.assign(kind=kind, expiry=expiry))
    out = pd.concat(frames, ignore_index=True)
    out["spot"] = spot
    out["snapshot_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    path = f"{ticker.lower()}_options_{today}.csv"
    out.to_csv(path, index=False)
    print(
        f"saved {len(out)} quotes across {out.expiry.nunique()} expiries to {path}; spot {spot:.2f}"
    )
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default="SPY")
    snapshot(parser.parse_args().ticker)
