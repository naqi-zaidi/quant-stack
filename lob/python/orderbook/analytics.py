"""Microstructure analytics computed from the order book's event stream (lob/analytics.hpp).

Each class is a C++ listener: attach it to a book or replay before the first event, and it
rebuilds what it needs from the events alone. Handlers update state on every event; the
methods compute values on demand. Time-weighted statistics charge each state for the time it
lasted, so pass `now` (ns) to include the interval since the last event.

    >>> from orderbook import ArrayOrderBook, Side
    >>> from orderbook.analytics import OrderFlowImbalance, SpreadDistribution
    >>> book = ArrayOrderBook()
    >>> spread, ofi = SpreadDistribution(), OrderFlowImbalance(window=50)
    >>> book.attach(spread); book.attach(ofi)
    >>> _ = book.submit(1, Side.Buy, 99, 10, ts=0)
    >>> _ = book.submit(2, Side.Sell, 101, 5, ts=10)
    >>> spread.histogram(now=30)  # 20 ns at a 2-tick spread
    [(2, 20)]
    >>> ofi.total, ofi.imbalance  # +10 (bid appears), -5 (ask appears)
    (5, 0.3333333333333333)
"""

from orderbook._lob import analytics as _analytics

DepthProfile = _analytics.DepthProfile
Listener = _analytics.Listener
NO_ASK = _analytics.NO_ASK
NO_BID = _analytics.NO_BID
OrderFlowImbalance = _analytics.OrderFlowImbalance
QueuePosition = _analytics.QueuePosition
QueuePositions = _analytics.QueuePositions
QueueStatus = _analytics.QueueStatus
Quote = _analytics.Quote
SpreadDistribution = _analytics.SpreadDistribution

__all__ = [
    "NO_ASK",
    "NO_BID",
    "DepthProfile",
    "Listener",
    "OrderFlowImbalance",
    "QueuePosition",
    "QueuePositions",
    "QueueStatus",
    "Quote",
    "SpreadDistribution",
]
