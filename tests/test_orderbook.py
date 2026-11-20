"""The C++ order book through its Python module (Project 2 step 5)."""

from __future__ import annotations

import doctest
import gc
import pickle
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import orderbook
import orderbook.gate as gate
import orderbook.text
from marketdata.lobster import EMPTY_ASK_PRICE, EMPTY_BID_PRICE
from orderbook import (
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
from orderbook import lobster as lb
from orderbook.text import book_text, event_text

BOOKS = [OrderBook, ArrayOrderBook]
REPLAYS = [lb.Replay, lb.ArrayReplay]
B, S = Side.Buy, Side.Sell
U64_MAX = 2**64 - 1


# ---------------------------------------------------------------- cross-language snapshots


@pytest.mark.parametrize("book_cls", BOOKS, ids=["map", "array"])
@pytest.mark.parametrize("path", sorted(gate.GOLDEN.glob("*.txt")), ids=lambda p: p.stem)
def test_golden_transcripts_rerun_from_python(path: Path, book_cls) -> None:
    expected = path.read_bytes().decode()
    if path.stem == "lobster_day":
        replay = lb.Replay if book_cls is OrderBook else lb.ArrayReplay
        actual, _, _ = gate.lobster_day_text(replay)
    else:
        actual = gate.rerun_transcript(expected, book_cls)
    assert actual == expected


@pytest.mark.parametrize("replay_cls", REPLAYS, ids=["map", "array"])
def test_lobster_day_hashes_to_its_cpp_pin(replay_cls) -> None:
    digest, events, py_digest = gate.pinned_day_digest(replay_cls)
    assert (digest, events) == gate.pinned_digest("lobster/consistent_day")
    assert py_digest == digest


def test_python_fnv_matches_published_vectors() -> None:
    def fnv(data: bytes) -> int:
        h = gate.FNV_OFFSET
        for byte in data:
            h = ((h ^ byte) * gate.FNV_PRIME) & gate.MASK64
        return h

    assert fnv(b"") == 0xCBF29CE484222325
    assert fnv(b"a") == 0xAF63DC4C8601EC8C
    assert fnv(b"foobar") == 0x85944171F73967E8


# ---------------------------------------------------------------- calls, events, views


@pytest.mark.parametrize("book_cls", BOOKS)
def test_sweep_trades_at_maker_prices_and_rests_the_rest(book_cls) -> None:
    book = book_cls()
    for oid, side, px, qty in gate.SEED_ORDERS:
        book.submit(oid, side, px, qty)
    book.take_events()
    r = book.submit(10, B, 102, 70, ts=5)
    assert r == Result((True, Reason.None_, 60, 10, 0))
    trades = [e for e in book.take_events() if isinstance(e, Trade)]
    assert [(t.maker, t.price, t.qty) for t in trades] == [(1, 101, 10), (2, 101, 20), (3, 102, 30)]
    assert all(t.taker == 10 and t.taker_side is B and t.ts == 5 for t in trades)
    assert book.best_bid() == 102 and book.best_ask() == 103
    assert book.best(B) == LevelView((102, 10, 1))
    assert book.depth(S) == [LevelView((103, 40, 1))]
    assert book.queue(B, 99) == [5, 6]
    assert book.find(10) == OrderView((10, B, OrderType.Limit, 102, 10, 5))
    assert book.find(1) is None
    assert (len(book), book.order_count) == (5, 5)
    assert (book.level_count(B), book.resting_qty(B)) == (3, 70)
    assert book.validate() == ""


@pytest.mark.parametrize("book_cls", BOOKS)
def test_every_order_type_and_rejection_reaches_python(book_cls) -> None:
    book = book_cls(tick_size=1)
    book.submit(1, S, 101, 10)
    assert book.submit(2, B, 0, 4, OrderType.Market).filled == 4
    assert book.submit(3, B, 101, 10, OrderType.IOC) == Result((True, Reason.None_, 6, 0, 4))
    assert book.submit(4, B, 101, 1, OrderType.FOK).reason is Reason.InsufficientLiquidity
    book.submit(5, S, 101, 5)
    assert book.submit(6, B, 101, 1, OrderType.PostOnly).reason is Reason.WouldCross
    assert book.submit(5, S, 102, 1).reason is Reason.DuplicateOrderId
    assert book.submit(7, S, 102, 0).reason is Reason.InvalidQuantity
    assert book.submit(8, S, -1, 1).reason is Reason.InvalidPrice
    assert book.cancel(999).reason is Reason.UnknownOrder
    assert book.modify(5, 101, 2) == Result((True, Reason.None_, 0, 2, 0))
    expired = [e for e in book.take_events() if isinstance(e, ExecReport) and e.reason]
    assert {e.reason for e in expired} >= {Reason.Unfilled, Reason.WouldCross}


def test_tick_grid_and_bad_arguments() -> None:
    book = ArrayOrderBook(tick_size=100)
    assert book.tick_size == 100
    assert book.submit(1, B, 5_853_350, 1).reason is Reason.InvalidPrice  # off the cent grid
    with pytest.raises(ValueError):
        ArrayOrderBook(tick_size=0)
    for bad in [
        lambda: book.submit(-1, B, 100, 1),  # negative id: ids are unsigned
        lambda: book.submit(2**64, B, 100, 1),  # id beyond 64 bits
        lambda: book.submit(2, 0, 100, 1),  # plain int for a Side
        lambda: book.submit(2, OrderType.Limit, 100, 1),  # wrong enum
        lambda: book.submit(2, B, 100.5, 1),  # float price
        lambda: book.submit(2, B, 100, 1, 0),  # plain int for an OrderType
    ]:
        with pytest.raises(TypeError):
            bad()
    with pytest.raises(TypeError):
        OrderBookView()  # views come only from a replay


@pytest.mark.parametrize("book_cls", BOOKS)
def test_unsigned_extremes_survive_the_round_trip(book_cls) -> None:
    book = book_cls()
    book.submit(U64_MAX, S, 2**40, 2**40, ts=U64_MAX)
    book.submit(U64_MAX - 1, B, 2**40, 1, OrderType.IOC, ts=U64_MAX)
    trade = next(e for e in book.take_events() if isinstance(e, Trade))
    assert (trade.taker, trade.maker, trade.ts) == (U64_MAX - 1, U64_MAX, U64_MAX)
    assert book.find(U64_MAX).qty == 2**40 - 1


def test_recording_and_digest_switches() -> None:
    full = ArrayOrderBook()
    quiet = ArrayOrderBook(record_events=False)
    bare = ArrayOrderBook(record_events=False, digest_events=False)
    for book in (full, quiet, bare):
        book.submit(1, S, 101, 10)
        book.submit(2, B, 101, 4)
    events = full.take_events()
    assert quiet.take_events() == [] and bare.take_events() == []
    assert full.digest == quiet.digest == gate.fnv1a64(events)
    assert bare.digest is None
    assert full.event_count == quiet.event_count == bare.event_count == len(events) == 7
    quiet.record_events = True
    quiet.cancel(1)
    assert [e.type for e in quiet.take_events() if isinstance(e, ExecReport)] == [
        ExecType.Cancelled
    ]


def test_value_types_are_named_tuples() -> None:
    book = ArrayOrderBook()
    r = book.submit(1, S, 101, 10)
    accepted, reason, *_ = r
    assert isinstance(r, tuple) and accepted and reason is Reason.None_
    events = book.take_events()
    match events[1]:
        case LevelUpdate(side=side, price=price, qty=qty):
            assert (side, price, qty) == (S, 101, 10)
        case _:
            pytest.fail("expected a LevelUpdate")
    assert pickle.loads(pickle.dumps(events)) == events
    assert pickle.loads(pickle.dumps(Side.Sell)) is Side.Sell
    assert event_text(events[0]) == "report id=1 New Sell px=101 last=0 leaves=10 ts=0"
    assert book_text(book) == book.to_text()
    assert [m.name for m in Reason][:2] == ["None_", "DuplicateOrderId"]


@pytest.mark.parametrize("module", [orderbook, orderbook.text, lb], ids=lambda m: m.__name__)
def test_docstring_examples_run(module) -> None:
    assert doctest.testmod(module).failed == 0


# ---------------------------------------------------------------- property: map == array


op = st.one_of(
    st.tuples(
        st.just("submit"),
        st.integers(1, 25),
        st.sampled_from(list(Side)),
        st.integers(-1, 12),
        st.integers(-1, 30),
        st.sampled_from(list(OrderType)),
    ),
    st.tuples(st.just("cancel"), st.integers(1, 25)),
    st.tuples(st.just("modify"), st.integers(1, 25), st.integers(-1, 12), st.integers(-1, 30)),
)


@given(st.lists(op, max_size=60))
def test_both_books_agree_on_any_sequence(ops) -> None:
    books = [OrderBook(), ArrayOrderBook()]
    for ts, call in enumerate(ops, start=1):
        outcomes = []
        for book in books:
            if call[0] == "submit":
                _, oid, side, px, qty, kind = call
                r = book.submit(oid, side, 100 + px, qty, kind, ts)
                if r.accepted:
                    assert r.filled + r.resting + r.cancelled == qty
            elif call[0] == "cancel":
                r = book.cancel(call[1], ts)
            else:
                r = book.modify(call[1], 100 + call[2], call[3], ts)
            outcomes.append((r, book.take_events()))
            bid, ask = book.best_bid(), book.best_ask()
            assert bid is None or ask is None or bid < ask
        assert outcomes[0] == outcomes[1]
    for book in books:
        assert book.validate() == ""
    assert books[0].digest == books[1].digest
    assert [books[0].orders(s) for s in Side] == [books[1].orders(s) for s in Side]


# ---------------------------------------------------------------- LOBSTER files


def write_day(folder: Path, levels: int = 2) -> Path:
    """The five-event day as LOBSTER files (the same day as tests/test_lobster.py)."""
    stem = f"TEST_2012-06-21_34200000_57600000_{{}}_{levels}.csv"
    msg_rows, book_rows = [], []
    for k, (m, snap) in enumerate(zip(gate.LOBSTER_DAY, gate.LOBSTER_BOOKS, strict=True)):
        direction = 1 if m.side is B else -1
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


@pytest.mark.parametrize("replay_cls", REPLAYS, ids=["map", "array"])
def test_replay_a_day_from_files(tmp_path: Path, replay_cls) -> None:
    msg = write_day(tmp_path)
    info = lb.describe_file(str(msg))
    assert (info.ticker, info.date, info.levels) == ("TEST", "2012-06-21", 2)

    rp = replay_cls(info.levels, 100)
    stats = rp.run(str(msg))
    assert (stats.events, stats.mismatches, stats.errors) == (4, 0, 0)
    assert stats.by_type[int(lb.Event.ExecuteVisible)] == 1
    assert stats.executions_matched == 1
    with pytest.raises(RuntimeError):
        rp.run(str(msg))  # a replay runs once

    # The same day one event at a time, driven from Python, gives the same stream.
    stepped = replay_cls(info.levels, 100, record_events=True)
    rows = list(lb.DayReader(str(msg)))
    assert len(rows) == 5 and rows[3][0].event is lb.Event.ExecuteVisible
    first, snap0 = rows[0]
    assert first.ns == 34_200_001_000_000 and snap0 == gate.LOBSTER_BOOKS[0]
    stepped.seed(snap0, first.ns)
    assert all(stepped.step(m, s) for m, s in rows[1:])
    assert stepped.digest == rp.digest == gate.fnv1a64(stepped.take_events())
    view = stepped.book
    assert isinstance(view, ArrayOrderBookView if replay_cls is lb.ArrayReplay else OrderBookView)
    assert (view.best_bid(), view.best_ask()) == (5853300, 5853500)


def test_replay_book_view_keeps_its_replay_alive() -> None:
    rp = lb.ArrayReplay(2, 100)
    rp.seed(gate.LOBSTER_BOOKS[0], 1)
    view = rp.book
    del rp
    gc.collect()
    assert view.best_bid() == 5853300


def test_parse_lines_and_bad_files(tmp_path: Path) -> None:
    m = lb.parse_message("34200.004,4,2,10,5853500,-1")
    assert (m.ns, m.event, m.id, m.size, m.price, m.side) == (
        34_200_004_000_000, lb.Event.ExecuteVisible, 2, 10, 5853500, S,
    )
    snap = lb.parse_snapshot(f"5853500,40,5853300,100,{EMPTY_ASK_PRICE},0,{EMPTY_BID_PRICE},0", 2)
    assert snap == lb.Snapshot(asks=[(5853500, 40)], bids=[(5853300, 100)])
    with pytest.raises(RuntimeError):
        lb.parse_message("not,a,lobster,line")
    with pytest.raises(RuntimeError):
        lb.ArrayReplay(2, 100).run(str(tmp_path / "TEST_2012-06-21_1_2_message_2.csv"))
