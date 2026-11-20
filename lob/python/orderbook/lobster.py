"""LOBSTER files and replay, backed by the C++ reader and replay (lob/lobster.hpp).

    >>> from orderbook.lobster import ArrayReplay
    >>> rp = ArrayReplay(levels=10, tick=100)
    >>> stats = rp.run("data/AAPL_2012-06-21_34200000_57600000_message_10.csv")  # doctest: +SKIP
    >>> stats.mismatches  # doctest: +SKIP
    0

run() replays the whole day in C++ with the GIL released. seed() and step() go one event at a
time, with DayReader iterating the two files as (Message, Snapshot) pairs.
"""

from orderbook._lob import lobster as _lobster

ArrayReplay = _lobster.ArrayReplay
DayReader = _lobster.DayReader
Event = _lobster.Event
FileInfo = _lobster.FileInfo
Message = _lobster.Message
Replay = _lobster.Replay
ReplayStats = _lobster.ReplayStats
Snapshot = _lobster.Snapshot
describe_file = _lobster.describe_file
parse_message = _lobster.parse_message
parse_snapshot = _lobster.parse_snapshot

__all__ = [
    "ArrayReplay",
    "DayReader",
    "Event",
    "FileInfo",
    "Message",
    "Replay",
    "ReplayStats",
    "Snapshot",
    "describe_file",
    "parse_message",
    "parse_snapshot",
]
