# data/

Market data lives here locally and is never committed (see `.gitignore`).

## LOBSTER (Nasdaq order book reconstruction)

1. Download a free sample from the [LOBSTER sample page](https://lobsterdata.com/info/DataSamples.php)
   (for example AAPL, 10 levels).
2. Unzip it into this folder. You should see a matching pair:

   ```
   AAPL_2012-06-21_34200000_57600000_message_10.csv
   AAPL_2012-06-21_34200000_57600000_orderbook_10.csv
   ```

3. Load it:

   ```python
   from marketdata import load_day, mid_price

   day = load_day("data/AAPL_2012-06-21_34200000_57600000_message_10.csv")
   day.messages.head()  # time, event, order_id, size, price, direction
   mid_price(day.book)  # level-1 mid in dollars
   ```

`load_day` checks that the two files line up row for row, that timestamps never go backwards,
and that the reference book is never crossed. A file that fails any check raises
`LobsterFormatError` rather than loading silently.

## SPY option-chain snapshot (committed)

`snapshots/spy_options_2026-11-19.csv.gz` is the yfinance snapshot taken 2026-11-19 at
10:03 New York time (8,413 quotes, 24 expiries). The step 6 Heston calibration uses it. It is
committed (gzipped, about 290 KB) so the calibration reproduces exactly. yfinance is an
unofficial scraper, and re-downloading would give a different market.

To take a fresh one, during US market hours:

```
python -m pip install --user yfinance
python -m marketdata.yfinance_snapshot SPY
```

`marketdata.option_chain.build_surface` turns a snapshot into a clean implied-vol surface:
parity forwards, OTM quotes only, quote filters and Black-76 vols.

## Coming with later phases

- Crypto exchange L2 websocket snapshots (Project 3, live order books for the market maker)
