"""Python interface to the C++20 limit order book (quant-stack Project 2).

Prices are integer ticks (LOBSTER units: dollars x 10,000), never floats.

    >>> from orderbook import ArrayOrderBook, OrderType, Side
    >>> book = ArrayOrderBook(tick_size=100)
    >>> book.submit(1, Side.Sell, 5_853_500, 100)
    orderbook.Result(accepted=True, reason=<Reason.None_: 0>, filled=0, resting=100, cancelled=0)
    >>> book.submit(2, Side.Buy, 5_853_500, 30, OrderType.IOC).filled
    30
    >>> book.take_events()[3]
    orderbook.Trade(taker=2, maker=1, taker_side=<Side.Buy: 0>, price=5853500, qty=30, ts=0)

OrderBook keeps price levels in a std::map; ArrayOrderBook in a tick-indexed array (faster,
allocation-free). Both emit identical event streams. Events, results and views are named
tuples. orderbook.text prints them as the C++ does; LOBSTER replay is in orderbook.lobster.
"""

from orderbook._lob import (
    ArrayOrderBook,
    ArrayOrderBookView,
    ExecReport,
    ExecType,
    LevelUpdate,
    LevelView,
    OrderBook,
    OrderBookView,
    OrderType,
    OrderView,
    Reason,
    Result,
    Side,
    Trade,
)

Event = Trade | LevelUpdate | ExecReport

__all__ = [
    "ArrayOrderBook",
    "ArrayOrderBookView",
    "Event",
    "ExecReport",
    "ExecType",
    "LevelUpdate",
    "LevelView",
    "OrderBook",
    "OrderBookView",
    "OrderType",
    "OrderView",
    "Reason",
    "Result",
    "Side",
    "Trade",
]
