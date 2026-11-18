"""LOBSTER loader tests on a hand-built five-event day with a 2-level book."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from marketdata import Event, LobsterFormatError, load_day, mid_price, parse_filename
from marketdata.lobster import EMPTY_ASK_PRICE as EA
from marketdata.lobster import EMPTY_BID_PRICE as EB

STEM = "TEST_2012-06-21_34200000_57600000"

# time, event, order_id, size, price, direction
MESSAGES = [
    (34200.001, 1, 1, 100, 5853300, 1),  # buy 100 @ 585.33
    (34200.002, 1, 2, 50, 5853500, -1),  # sell 50 @ 585.35
    (34200.003, 1, 3, 20, 5853200, 1),  # buy 20 @ 585.32
    (34200.004, 4, 2, 10, 5853500, -1),  # 10 of order 2 executes
    (34200.005, 3, 3, 20, 5853200, 1),  # order 3 deleted
]

# ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ask_s2, bid_p2, bid_s2 after each message
BOOK = [
    (EA, 0, 5853300, 100, EA, 0, EB, 0),
    (5853500, 50, 5853300, 100, EA, 0, EB, 0),
    (5853500, 50, 5853300, 100, EA, 0, 5853200, 20),
    (5853500, 40, 5853300, 100, EA, 0, 5853200, 20),
    (5853500, 40, 5853300, 100, EA, 0, EB, 0),
]


def _write(path: Path, rows) -> None:
    path.write_text("\n".join(",".join(str(v) for v in r) for r in rows) + "\n")


@pytest.fixture
def day_dir(tmp_path: Path) -> Path:
    _write(tmp_path / f"{STEM}_message_2.csv", MESSAGES)
    _write(tmp_path / f"{STEM}_orderbook_2.csv", BOOK)
    return tmp_path


def test_parse_filename():
    meta, kind = parse_filename("AAPL_2012-06-21_34200000_57600000_message_10.csv")
    assert kind == "message"
    assert meta.ticker == "AAPL"
    assert meta.date == date(2012, 6, 21)
    assert (meta.start_ms, meta.end_ms, meta.levels) == (34_200_000, 57_600_000, 10)


@pytest.mark.parametrize("name", ["AAPL_message_10.csv", "aapl_2012-06-21_1_2_message_10.csv"])
def test_parse_filename_rejects_other_names(name):
    with pytest.raises(LobsterFormatError):
        parse_filename(name)


def test_load_day_shapes_and_types(day_dir):
    day = load_day(day_dir / f"{STEM}_message_2.csv")
    assert len(day) == len(MESSAGES)
    assert day.meta.levels == 2
    assert list(day.messages.columns) == ["time", "event", "order_id", "size", "price", "direction"]
    assert day.messages["price"].dtype == "int64"
    assert day.book.shape == (5, 8)
    assert (day.messages["event"].iloc[3]) == Event.EXECUTE_VISIBLE


def test_empty_levels_load_as_missing(day_dir):
    book = load_day(day_dir / f"{STEM}_message_2.csv").book
    assert pd.isna(book.loc[0, "ask_price_1"])
    assert pd.isna(book.loc[0, "ask_size_1"])
    assert pd.isna(book.loc[4, "bid_price_2"])
    assert book.loc[2, "bid_price_2"] == 5853200


def test_mid_price(day_dir):
    mid = mid_price(load_day(day_dir / f"{STEM}_message_2.csv").book)
    assert pd.isna(mid.iloc[0])  # no ask yet
    assert mid.iloc[1] == pytest.approx(585.34)


def test_row_count_mismatch_is_rejected(day_dir):
    _write(day_dir / f"{STEM}_orderbook_2.csv", BOOK[:-1])
    with pytest.raises(LobsterFormatError, match="row count"):
        load_day(day_dir / f"{STEM}_message_2.csv")


def test_unknown_event_is_rejected(day_dir):
    bad = list(MESSAGES)
    bad[2] = (34200.003, 9, 3, 20, 5853200, 1)
    _write(day_dir / f"{STEM}_message_2.csv", bad)
    with pytest.raises(LobsterFormatError, match="event type"):
        load_day(day_dir / f"{STEM}_message_2.csv")


def test_time_going_backwards_is_rejected(day_dir):
    bad = list(MESSAGES)
    bad[3] = (34200.0001, *MESSAGES[3][1:])
    _write(day_dir / f"{STEM}_message_2.csv", bad)
    with pytest.raises(LobsterFormatError, match="backwards"):
        load_day(day_dir / f"{STEM}_message_2.csv")


def test_crossed_book_is_rejected(day_dir):
    bad = list(BOOK)
    bad[1] = (5853300, 50, 5853300, 100, EA, 0, EB, 0)  # bid == ask
    _write(day_dir / f"{STEM}_orderbook_2.csv", bad)
    with pytest.raises(LobsterFormatError, match="crossed"):
        load_day(day_dir / f"{STEM}_message_2.csv")


def test_wrong_column_count_is_rejected(day_dir):
    _write(day_dir / f"{STEM}_orderbook_2.csv", [r[:4] for r in BOOK])
    with pytest.raises(LobsterFormatError, match="columns"):
        load_day(day_dir / f"{STEM}_message_2.csv")
