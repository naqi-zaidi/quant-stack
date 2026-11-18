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

## Coming with later phases

- Crypto exchange L2 websocket snapshots (Project 3, live order books for the market maker)
- SPY option chain snapshots, cached to disk (Project 1, Heston calibration)
