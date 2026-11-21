from pathlib import Path

import pytest
from hypothesis import settings

import orderbook.gate as gate
from marketdata.lobster import EMPTY_ASK_PRICE, EMPTY_BID_PRICE
from orderbook import Side

# Derandomized so every CI run explores the same examples: test results reproduce exactly.
settings.register_profile("default", derandomize=True, deadline=None, max_examples=300)
settings.load_profile("default")


def write_day(folder: Path, levels: int = 2) -> Path:
    """The five-event day as LOBSTER files (the same day as tests/test_lobster.py)."""
    stem = f"TEST_2012-06-21_34200000_57600000_{{}}_{levels}.csv"
    msg_rows, book_rows = [], []
    for k, (m, snap) in enumerate(zip(gate.LOBSTER_DAY, gate.LOBSTER_BOOKS, strict=True)):
        direction = 1 if m.side is Side.Buy else -1
        msg_rows.append(f"34200.{k + 1:03d},{int(m.event)},{m.id},{m.size},{m.price},{direction}")
        cells = []
        for lvl in range(levels):
            ask = snap.asks[lvl] if lvl < len(snap.asks) else (EMPTY_ASK_PRICE, 0)
            bid = snap.bids[lvl] if lvl < len(snap.bids) else (EMPTY_BID_PRICE, 0)
            cells += [*ask, *bid]
        book_rows.append(",".join(map(str, cells)))
    msg = folder / stem.format("message")
    msg.write_text("\n".join(msg_rows) + "\n")
    (folder / stem.format("orderbook")).write_text("\n".join(book_rows) + "\n")
    return msg


@pytest.fixture
def lobster_day(tmp_path: Path) -> Path:
    """Message file of the five-event day, with its orderbook file beside it."""
    return write_day(tmp_path)
