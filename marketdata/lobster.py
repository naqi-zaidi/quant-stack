"""Loader for LOBSTER limit order book data (https://lobsterdata.com).

A LOBSTER day ships as two CSV files with no header and the same number of rows:

* message file: one row per book event
  ``time, event_type, order_id, size, price, direction``
* orderbook file: the book state *after* that event, ``levels`` deep, as repeating
  ``ask_price_i, ask_size_i, bid_price_i, bid_size_i`` groups

Row k of the orderbook is the reference the C++ book is reconciled against after
applying message k (the Phase 2 gate).

Prices are integers in units of $0.0001 (dollars * 10_000) and are kept as integers here.
Empty book levels are written as price 9999999999 (ask) or -9999999999 (bid) with size 0;
they load as missing values so nothing downstream mistakes a sentinel for a real quote.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_SCALE = 10_000
EMPTY_ASK_PRICE = 9_999_999_999
EMPTY_BID_PRICE = -9_999_999_999

MESSAGE_COLUMNS = ["time", "event", "order_id", "size", "price", "direction"]

_FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z.]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<start>\d+)_(?P<end>\d+)_"
    r"(?P<kind>message|orderbook)_(?P<levels>\d+)\.csv$"
)


class Event(IntEnum):
    """LOBSTER event types (message file, column 2)."""

    SUBMIT = 1  # new limit order
    CANCEL = 2  # partial cancellation, size reduced
    DELETE = 3  # full deletion of a limit order
    EXECUTE_VISIBLE = 4  # execution against a visible resting order
    EXECUTE_HIDDEN = 5  # execution against a hidden order (does not change the visible book)
    CROSS = 6  # auction cross trade
    HALT = 7  # trading halt indicator; price column: -1 halt, 0 quoting, 1 resume


class Direction(IntEnum):
    """Side of the *resting* order. For executions, BUY means a sell order hit the bid."""

    SELL = -1
    BUY = 1


class LobsterFormatError(ValueError):
    """Raised when a file does not match the LOBSTER layout."""


@dataclass(frozen=True)
class LobsterMeta:
    ticker: str
    date: date
    start_ms: int  # milliseconds after midnight, e.g. 34_200_000 = 09:30
    end_ms: int
    levels: int


@dataclass(frozen=True)
class LobsterDay:
    meta: LobsterMeta
    messages: pd.DataFrame
    book: pd.DataFrame

    def __len__(self) -> int:
        return len(self.messages)


def parse_filename(path: str | Path) -> tuple[LobsterMeta, str]:
    """Parse ``AAPL_2012-06-21_34200000_57600000_message_10.csv`` into metadata and file kind."""
    name = Path(path).name
    m = _FILENAME_RE.match(name)
    if m is None:
        raise LobsterFormatError(f"not a LOBSTER file name: {name!r}")
    meta = LobsterMeta(
        ticker=m["ticker"],
        date=date.fromisoformat(m["date"]),
        start_ms=int(m["start"]),
        end_ms=int(m["end"]),
        levels=int(m["levels"]),
    )
    return meta, m["kind"]


def book_columns(levels: int) -> list[str]:
    cols: list[str] = []
    for i in range(1, levels + 1):
        cols += [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
    return cols


def read_messages(path: str | Path) -> pd.DataFrame:
    """Read a message file into typed columns and validate it.

    ``time`` is seconds after midnight (float64, nanosecond precision in the source).
    ``price`` stays an integer in $0.0001 units; divide by ``PRICE_SCALE`` for dollars.
    """
    df = pd.read_csv(
        path,
        header=None,
        usecols=range(6),
        names=MESSAGE_COLUMNS,
        dtype={
            "time": np.float64,
            "event": np.int8,
            "order_id": np.int64,
            "size": np.int64,
            "price": np.int64,
            "direction": np.int8,
        },
    )
    _validate_messages(df, path)
    return df


def read_orderbook(path: str | Path, levels: int | None = None) -> pd.DataFrame:
    """Read an orderbook file. Empty levels become <NA> in nullable Int64 columns."""
    if levels is None:
        levels = parse_filename(path)[0].levels
    raw = pd.read_csv(path, header=None, dtype=np.int64)
    if raw.shape[1] != 4 * levels:
        raise LobsterFormatError(
            f"{Path(path).name}: expected {4 * levels} columns for {levels} levels, "
            f"got {raw.shape[1]}"
        )
    raw.columns = book_columns(levels)
    book = raw.astype("Int64")
    for i in range(1, levels + 1):
        ask_empty = raw[f"ask_price_{i}"] == EMPTY_ASK_PRICE
        bid_empty = raw[f"bid_price_{i}"] == EMPTY_BID_PRICE
        book.loc[ask_empty, [f"ask_price_{i}", f"ask_size_{i}"]] = pd.NA
        book.loc[bid_empty, [f"bid_price_{i}", f"bid_size_{i}"]] = pd.NA
    _validate_book(book, levels, path)
    return book


def load_day(message_path: str | Path) -> LobsterDay:
    """Load a message file and its matching orderbook file from the same folder."""
    message_path = Path(message_path)
    meta, kind = parse_filename(message_path)
    if kind != "message":
        raise LobsterFormatError(f"expected a message file, got {message_path.name!r}")
    book_path = message_path.with_name(message_path.name.replace("_message_", "_orderbook_"))
    messages = read_messages(message_path)
    book = read_orderbook(book_path, meta.levels)
    if len(messages) != len(book):
        raise LobsterFormatError(
            f"row count mismatch: {len(messages)} messages vs {len(book)} book snapshots"
        )
    return LobsterDay(meta=meta, messages=messages, book=book)


def mid_price(book: pd.DataFrame) -> pd.Series:
    """Level-1 mid in dollars; <NA> when either side of the book is empty."""
    return (book["ask_price_1"] + book["bid_price_1"]) / (2 * PRICE_SCALE)


def _validate_messages(df: pd.DataFrame, path: str | Path) -> None:
    name = Path(path).name
    bad_events = ~df["event"].isin([int(e) for e in Event])
    if bad_events.any():
        row = int(np.flatnonzero(bad_events.to_numpy())[0])
        raise LobsterFormatError(f"{name}: unknown event type at row {row}")
    bad_dir = ~df["direction"].isin([int(d) for d in Direction])
    if bad_dir.any():
        row = int(np.flatnonzero(bad_dir.to_numpy())[0])
        raise LobsterFormatError(f"{name}: direction must be -1 or 1, row {row}")
    if (np.diff(df["time"].to_numpy()) < 0).any():
        raise LobsterFormatError(f"{name}: timestamps go backwards")


def _validate_book(book: pd.DataFrame, levels: int, path: str | Path) -> None:
    name = Path(path).name
    # A resting book is never crossed: best bid < best ask whenever both exist.
    ask, bid = book["ask_price_1"], book["bid_price_1"]
    crossed = (bid >= ask).fillna(False)
    if crossed.any():
        row = int(np.flatnonzero(crossed.to_numpy())[0])
        raise LobsterFormatError(f"{name}: crossed book at row {row}")
    # Within a side, prices are strictly ordered from level 1 outward.
    for i in range(1, levels):
        worse_ask = (book[f"ask_price_{i + 1}"] <= book[f"ask_price_{i}"]).fillna(False)
        worse_bid = (book[f"bid_price_{i + 1}"] >= book[f"bid_price_{i}"]).fillna(False)
        if worse_ask.any() or worse_bid.any():
            raise LobsterFormatError(f"{name}: levels out of order at depth {i + 1}")
