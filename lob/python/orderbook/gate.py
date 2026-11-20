"""Phase 2 step 5 gate: through the Python module, the engine behaves exactly as it does in C++.

1. Golden transcripts. Each scripted scenario in lob/tests/golden/ is parsed back into its
   calls and re-run from Python on both books. Python's own formatter (below, written
   independently of text.hpp) prints the result, which must equal the file byte for byte.
   The LOBSTER-day transcript is rebuilt with orderbook.lobster.ArrayReplay and Replay.
2. Pinned digest. On both books, the LOBSTER day's event stream as received in Python hashes
   to the digest the C++ tests pin for it (lobster/consistent_day in pinned_digests.hpp).
   The hash is a Python FNV-1a implementation.
3. Random flows. Five fixed-seed regimes go through both books. Results and events must agree
   map vs array, and Python's FNV-1a over the received events must equal the C++ digest.
   Every accepted order must conserve quantity; the book must never cross; validate() is run
   periodically.
4. Local only, when a LOBSTER day is in data/. Replayed from Python both ways (run() in C++,
   and step() per event driven by a Python loop), on both books, it must give 0 mismatches,
   0 errors and one digest. For the AAPL sample, that digest must equal lob_replay's.

Not gated, printed only: what a call costs from Python. Timings depend on the machine.

    uv run python -m orderbook.gate
    uv run python -m orderbook.gate --no-bench
"""

from __future__ import annotations

import argparse
import re
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from orderbook import (
    ArrayOrderBook,
    ExecReport,
    LevelUpdate,
    OrderBook,
    OrderType,
    Result,
    Side,
    Trade,
)
from orderbook import lobster as lb
from orderbook.text import book_text, events_text, result_text

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "lob" / "tests" / "golden"
PINS = ROOT / "lob" / "tests" / "pinned_digests.hpp"
DATA = ROOT / "data"

BOOKS = {"map": OrderBook, "array": ArrayOrderBook}
REPLAYS = {"map": lb.Replay, "array": lb.ArrayReplay}

# lob_replay's digest for the LOBSTER AAPL sample (README, step 3), so a Python replay of that
# file is held to the same number.
KNOWN_DAYS = {
    "AAPL_2012-06-21_34200000_57600000_message_10.csv": (0x089E41B222534921, 1_283_305),
}

# ---------------------------------------------------------------------------------------------
# Transcripts: the format of lob/tests/test_golden.cpp, driven from Python and printed by
# orderbook.text (a Python implementation of text.hpp, independent of it).

LOBSTER_EVENT_NAMES = {
    lb.Event.Submit: "submit",
    lb.Event.Cancel: "partial cancel",
    lb.Event.Delete: "delete",
    lb.Event.ExecuteVisible: "visible execution",
    lb.Event.ExecuteHidden: "hidden execution",
    lb.Event.Cross: "cross",
    lb.Event.Halt: "halt",
}


# The standard small book of test_golden.cpp's Script::seed(), built quietly at ts 0.
SEED_ORDERS = [
    (1, Side.Sell, 101, 10),
    (2, Side.Sell, 101, 20),
    (3, Side.Sell, 102, 30),
    (4, Side.Sell, 103, 40),
    (5, Side.Buy, 99, 10),
    (6, Side.Buy, 99, 20),
    (7, Side.Buy, 98, 30),
]


class Script:
    """Drives a book and transcribes it; each call gets the next timestamp."""

    def __init__(self, book_cls, title: str) -> None:
        self.book = book_cls(tick_size=1, expected_orders=64)
        self.out = [f"# {title}\n"]
        self.ts = 0

    def submit(self, oid: int, side: Side, kind: OrderType, px: int, qty: int) -> None:
        self.ts += 1
        call = f"submit #{oid} {side.name} {kind.name} {qty}"
        if kind != OrderType.Market:
            call += f" @ {px}"
        self._record(call, self.book.submit(oid, side, px, qty, kind, self.ts))

    def cancel(self, oid: int) -> None:
        self.ts += 1
        self._record(f"cancel #{oid}", self.book.cancel(oid, self.ts))

    def modify(self, oid: int, px: int, qty: int) -> None:
        self.ts += 1
        self._record(f"modify #{oid} to {qty} @ {px}", self.book.modify(oid, px, qty, self.ts))

    def seed(self) -> None:
        for oid, side, px, qty in SEED_ORDERS:
            self.book.submit(oid, side, px, qty)
        self.book.take_events()
        self.out.append("\nseeded book\n" + book_text(self.book))

    def show(self) -> None:
        text = book_text(self.book)
        if text != self.book.to_text():
            raise AssertionError("Python ladder differs from C++ to_text(book)")
        self.out.append("\nbook\n" + text)

    def text(self) -> str:
        return "".join(self.out)

    def _record(self, call: str, r: Result) -> None:
        events = events_text(self.book.take_events())
        self.out.append(f"\n> {call}\n  = {result_text(r)}\n{events}")


_SUBMIT = re.compile(r"> submit #(\d+) (Buy|Sell) (\w+) (-?\d+)(?: @ (-?\d+))?")
_CANCEL = re.compile(r"> cancel #(\d+)")
_MODIFY = re.compile(r"> modify #(\d+) to (-?\d+) @ (-?\d+)")


def rerun_transcript(golden: str, book_cls) -> str:
    """Parses a scripted transcript back into its calls and runs them again on `book_cls`."""
    lines = golden.split("\n")
    if not lines[0].startswith("# "):
        raise ValueError("a transcript starts with '# <title>'")
    script = Script(book_cls, lines[0][2:])
    for line in lines[1:]:
        if line == "seeded book":
            script.seed()
        elif line == "book":
            script.show()
        elif m := _SUBMIT.fullmatch(line):
            oid, side, kind, qty, px = m.groups()
            script.submit(int(oid), Side[side], OrderType[kind], int(px or 0), int(qty))
        elif m := _CANCEL.fullmatch(line):
            script.cancel(int(m[1]))
        elif m := _MODIFY.fullmatch(line):
            script.modify(int(m[1]), int(m[3]), int(m[2]))
        elif line.startswith(">"):
            raise ValueError(f"unrecognised call in transcript: {line!r}")
    return script.text()


# The five-event day of test_lobster.cpp (also lobster_day.txt and marketdata's tests).
B, S = Side.Buy, Side.Sell
LOBSTER_DAY = [
    lb.Message(ns=1, event=lb.Event.Submit, id=1, size=100, price=5853300, side=B),
    lb.Message(ns=2, event=lb.Event.Submit, id=2, size=50, price=5853500, side=S),
    lb.Message(ns=3, event=lb.Event.Submit, id=3, size=20, price=5853200, side=B),
    lb.Message(ns=4, event=lb.Event.ExecuteVisible, id=2, size=10, price=5853500, side=S),
    lb.Message(ns=5, event=lb.Event.Delete, id=3, size=20, price=5853200, side=B),
]
LOBSTER_BOOKS = [
    lb.Snapshot(asks=[], bids=[(5853300, 100)]),
    lb.Snapshot(asks=[(5853500, 50)], bids=[(5853300, 100)]),
    lb.Snapshot(asks=[(5853500, 50)], bids=[(5853300, 100), (5853200, 20)]),
    lb.Snapshot(asks=[(5853500, 40)], bids=[(5853300, 100), (5853200, 20)]),
    lb.Snapshot(asks=[(5853500, 40)], bids=[(5853300, 100)]),
]


def lobster_day_text(replay_cls) -> tuple[str, int, int]:
    """The lobster_day transcript, plus the replay's C++ digest and event count."""
    rp = replay_cls(2, 100, record_events=True)
    out = [
        "# LOBSTER replay of a five-event day (2 levels)\n",
        "# liquidity the replay seeds itself gets ids from 2^62 = 4611686018427387904 up\n\n",
        "> seed from the first snapshot\n",
    ]
    rp.seed(LOBSTER_BOOKS[0], LOBSTER_DAY[0].ns)
    out.append(events_text(rp.take_events()) + book_text(rp.book))
    for i in range(1, len(LOBSTER_DAY)):
        m = LOBSTER_DAY[i]
        ok = rp.step(m, LOBSTER_BOOKS[i])
        flag = "" if ok else "  (MISMATCH)"
        out.append(
            f"\n> event {i}: {LOBSTER_EVENT_NAMES[m.event]} #{m.id} {m.size} @ {m.price}{flag}\n"
            + events_text(rp.take_events())
            + book_text(rp.book)
        )
    return "".join(out), rp.digest, rp.event_count


def pinned_day_digest(replay_cls) -> tuple[int, int, int]:
    """The same day as test_lobster.cpp's Day replays it for its pin: every timestamp 0, tick 1.
    Returns (C++ digest, event count, Python FNV-1a over the events received)."""
    rp = replay_cls(2, record_events=True)
    rp.seed(LOBSTER_BOOKS[0])
    for m, book in zip(LOBSTER_DAY[1:], LOBSTER_BOOKS[1:], strict=True):
        msg = lb.Message(ns=0, event=m.event, id=m.id, size=m.size, price=m.price, side=m.side)
        rp.step(msg, book)
    return rp.digest, rp.event_count, fnv1a64(rp.take_events())


# ---------------------------------------------------------------------------------------------
# FNV-1a 64 over an event stream, byte for byte as digest.hpp: a tag byte, then each field as
# an 8-byte little-endian integer (signed values two's complement), in declaration order.

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fnv1a64(events: Iterable, h: int = FNV_OFFSET) -> int:
    for e in events:
        if isinstance(e, Trade):
            tag, fields = b"T", (e.taker, e.maker, e.taker_side, e.price, e.qty, e.ts)
        elif isinstance(e, LevelUpdate):
            tag, fields = b"L", (e.side, e.price, e.qty, e.count, e.ts)
        elif isinstance(e, ExecReport):
            tag = b"R"
            fields = (e.id, e.type, e.reason, e.side, e.price, e.last_qty, e.leaves, e.ts)
        else:
            raise TypeError(f"not an order book event: {e!r}")
        data = tag + b"".join((int(f) & MASK64).to_bytes(8, "little") for f in fields)
        for byte in data:
            h = ((h ^ byte) * FNV_PRIME) & MASK64
    return h


def pinned_digest(name: str) -> tuple[int, int]:
    """(digest, events) pinned for `name` in lob/tests/pinned_digests.hpp."""
    pattern = r'\{"' + re.escape(name) + r'",\s*0x([0-9a-fA-F]+)ULL,\s*(\d+)\}'
    m = re.search(pattern, PINS.read_text())
    if not m:
        raise KeyError(f"no pin named {name} in {PINS}")
    return int(m[1], 16), int(m[2])


# ---------------------------------------------------------------------------------------------
# Random flows.


class SplitMix64:
    """Portable 64-bit generator: the same seed gives the same flow on every platform."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def chance(self, per_thousand: int) -> bool:
        return self.below(1000) < per_thousand


@dataclass(frozen=True)
class Regime:
    name: str
    width: int  # prices drawn within +-width ticks of the mid
    aggressive: int  # per mille of submissions priced through the touch
    cancel: int  # per mille of calls that cancel
    modify: int  # per mille of calls that modify
    max_qty: int


REGIMES = [
    Regime("tight spread", 3, 250, 200, 100, 50),
    Regime("deep queues", 2, 100, 150, 150, 20),
    Regime("wide book", 200, 100, 200, 100, 100),
    Regime("cancel churn", 10, 100, 450, 250, 30),
    Regime("sweeps", 20, 450, 150, 50, 400),
]
MID = 100_000
KINDS = [OrderType.Limit] * 6 + [OrderType.IOC, OrderType.FOK, OrderType.PostOnly, OrderType.Market]


def flow(regime: Regime, seed: int, calls: int):
    """Yields ('submit', id, side, price, qty, type) / ('cancel', id) / ('modify', id, price,
    qty) with timestamps 1, 2, ... Includes invalid calls: zero quantities, duplicate and
    unknown ids, non-positive prices."""
    rng = SplitMix64(seed)
    next_id = 1
    recent: list[int] = []
    for ts in range(1, calls + 1):
        roll = rng.below(1000)
        if recent and roll < regime.cancel:
            yield ("cancel", recent[-1 - rng.below(min(len(recent), 64))], ts)
        elif recent and roll < regime.cancel + regime.modify:
            oid = recent[-1 - rng.below(min(len(recent), 64))]
            px = MID + rng.below(2 * regime.width + 1) - regime.width
            yield ("modify", oid, px, rng.below(regime.max_qty) + 1, ts)
        else:
            side = Side.Buy if rng.below(2) == 0 else Side.Sell
            offset = rng.below(regime.width + 1)
            through = rng.chance(regime.aggressive)
            # Passive: buys below the mid, sells above. Aggressive: across it.
            sign = -1 if (side == Side.Buy) != through else 1
            px = MID + sign * offset
            qty = rng.below(regime.max_qty) + 1
            kind = KINDS[rng.below(len(KINDS))]
            if rng.chance(5):
                qty = 0  # InvalidQuantity
            if rng.chance(5):
                px = -px  # InvalidPrice (ignored for market orders)
            oid = next_id
            if recent and rng.chance(10):
                oid = recent[-1]  # duplicate if still live
            else:
                next_id += 1
                recent.append(oid)
            yield ("submit", oid, side, px, qty, kind, ts)


@dataclass
class FlowReport:
    regime: str
    calls: int = 0
    accepted: int = 0
    trades: int = 0
    events: int = 0
    problems: int = 0
    first_problem: str = ""

    def problem(self, what: str) -> None:
        self.problems += 1
        if not self.first_problem:
            self.first_problem = what


def run_flow(regime: Regime, seed: int, calls: int, validate_every: int = 500) -> FlowReport:
    rep = FlowReport(regime.name)
    books = {k: cls(tick_size=1, expected_orders=4096) for k, cls in BOOKS.items()}
    hashes = dict.fromkeys(books, FNV_OFFSET)
    for i, call in enumerate(flow(regime, seed, calls), start=1):
        results, events = {}, {}
        for k, book in books.items():
            if call[0] == "submit":
                _, oid, side, px, qty, kind, ts = call
                results[k] = book.submit(oid, side, px, qty, kind, ts)
            elif call[0] == "cancel":
                results[k] = book.cancel(call[1], call[2])
            else:
                _, oid, px, qty, ts = call
                results[k] = book.modify(oid, px, qty, ts)
            events[k] = book.take_events()
            hashes[k] = fnv1a64(events[k], hashes[k])
        r = results["map"]
        rep.calls += 1
        rep.accepted += r.accepted
        rep.events += len(events["map"])
        rep.trades += sum(isinstance(e, Trade) for e in events["map"])
        if results["array"] != r:
            rep.problem(f"call {i} {call}: map {r} vs array {results['array']}")
        if events["array"] != events["map"]:
            rep.problem(f"call {i} {call}: event streams differ")
        if call[0] == "submit" and r.accepted and r.filled + r.resting + r.cancelled != call[4]:
            rep.problem(f"call {i} {call}: quantity not conserved, {r}")
        for k, book in books.items():
            bid, ask = book.best_bid(), book.best_ask()
            if bid is not None and ask is not None and bid >= ask:
                rep.problem(f"call {i}: {k} book crossed, bid {bid} >= ask {ask}")
            if i % validate_every == 0 and (err := book.validate()):
                rep.problem(f"call {i}: {k} book invalid: {err}")
    for k, book in books.items():
        if err := book.validate():
            rep.problem(f"end: {k} book invalid: {err}")
        if hashes[k] != book.digest:
            rep.problem(f"{k}: Python FNV-1a {hashes[k]:#018x} != C++ digest {book.digest:#018x}")
    for side in Side:
        if books["map"].orders(side) != books["array"].orders(side):
            rep.problem(f"end: {side.name} side differs map vs array")
    return rep


# ---------------------------------------------------------------------------------------------
# Real day (local only) and call cost (printed, not gated).


def find_lobster_day() -> Path | None:
    days = sorted(DATA.glob("*_message_*.csv")) if DATA.is_dir() else []
    return days[0] if days else None


def replay_day(path: Path) -> list[dict]:
    rows = []
    for name, cls in REPLAYS.items():
        info = lb.describe_file(str(path))
        rp = cls(info.levels, 100)
        t0 = time.perf_counter()
        st = rp.run(str(path))
        rows.append(
            {"how": f"{name}, run() in C++", "stats": st, "digest": rp.digest,
             "events": rp.event_count, "seconds": time.perf_counter() - t0}
        )
        rp = cls(info.levels, 100)
        t0 = time.perf_counter()
        reader = iter(lb.DayReader(str(path)))
        m, snap = next(reader)
        rp.seed(snap, m.ns)
        for m, snap in reader:
            rp.step(m, snap)
        rows.append(
            {"how": f"{name}, step() from a Python loop", "stats": rp.stats, "digest": rp.digest,
             "events": rp.event_count, "seconds": time.perf_counter() - t0}
        )
    return rows


def call_cost(book_cls, pairs: int = 100_000, repeats: int = 7, **book_kw) -> list[float]:
    """ns per call for submit + cancel pairs at a standing level. With events recorded, they are
    collected after every pair, as a strategy would."""
    out = []
    for _ in range(repeats):
        book = book_cls(tick_size=1, expected_orders=1024, **book_kw)
        for i in range(20):  # a background book so the level and its neighbours exist
            book.submit(1_000_000 + i, Side.Buy, MID - 1 - i % 5, 10)
            book.submit(2_000_000 + i, Side.Sell, MID + 1 + i % 5, 10)
        book.take_events()
        submit, cancel, take, buy = book.submit, book.cancel, book.take_events, Side.Buy
        recording = book.record_events
        t0 = time.perf_counter_ns()
        if recording:
            for oid in range(1, pairs + 1):
                submit(oid, buy, MID - 1, 5)
                cancel(oid)
                take()
        else:
            for oid in range(1, pairs + 1):
                submit(oid, buy, MID - 1, 5)
                cancel(oid)
        out.append((time.perf_counter_ns() - t0) / (2 * pairs))
    return out


def empty_loop_cost(pairs: int = 100_000, repeats: int = 7) -> list[float]:
    """The same loop calling a trivial Python function, for scale."""

    def nothing(*_args):
        return None

    out = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        for oid in range(1, pairs + 1):
            nothing(oid, 0, MID - 1, 5)
            nothing(oid)
        out.append((time.perf_counter_ns() - t0) / (2 * pairs))
    return out


def spread(xs: list[float]) -> str:
    return f"{statistics.median(xs):.0f} [{min(xs):.0f}–{max(xs):.0f}]"


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calls", type=int, default=10_000, help="calls per random-flow regime")
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--no-bench", action="store_true", help="skip the call-cost timing")
    ap.add_argument("--lobster", type=Path, help="LOBSTER message file (default: first in data/)")
    args = ap.parse_args(argv)
    failures: list[str] = []

    # 1-2. Golden transcripts and the pinned LOBSTER digest.
    print("## Golden transcripts re-run from Python\n")
    print("| Transcript | Map book | Array book |")
    print("|---|---|---|")
    files = sorted(GOLDEN.glob("*.txt"))
    scripted = [f for f in files if f.stem != "lobster_day"]
    pin, pin_events = pinned_digest("lobster/consistent_day")
    for f in files:
        expected = f.read_bytes().decode()
        cells = []
        for name in BOOKS:
            if f.stem == "lobster_day":
                actual, _, _ = lobster_day_text(REPLAYS[name])
            else:
                actual = rerun_transcript(expected, BOOKS[name])
            same = actual == expected
            cells.append("identical" if same else "**DIFFERS**")
            if not same:
                failures.append(f"golden {f.stem} on the {name} book")
        print(f"| {f.stem} | {cells[0]} | {cells[1]} |")
    if len(scripted) != 9 or len(files) != 10:
        failures.append(f"expected 10 golden files (9 scripted + lobster_day), found {len(files)}")
    for name, cls in REPLAYS.items():
        digest, n, py_digest = pinned_day_digest(cls)
        if not digest == py_digest == pin or n != pin_events:
            failures.append(
                f"lobster/consistent_day on {name}: C++ {digest:#x}, Python {py_digest:#x}, "
                f"{n} events; pinned {pin:#x}, {pin_events} events"
            )
    print(
        f"\nPinned digest: the same LOBSTER day replayed as test_lobster.cpp pins it (every "
        f"timestamp 0, tick 1), on both books, hashes to `{pin:#018x}` over {pin_events} events "
        "in C++ and in Python's FNV-1a over the events Python received: the value in "
        "pinned_digests.hpp (lobster/consistent_day)."
    )

    # 3. Random flows.
    print(f"\n## Random flows, map vs array, {args.calls:,} calls per regime, seed {args.seed}\n")
    print("| Regime | Calls | Accepted | Trades | Events | Problems |")
    print("|---|---:|---:|---:|---:|---:|")
    total = FlowReport("total")
    for j, regime in enumerate(REGIMES):
        rep = run_flow(regime, args.seed + j, args.calls)
        print(
            f"| {rep.regime} | {rep.calls:,} | {rep.accepted:,} | {rep.trades:,} | "
            f"{rep.events:,} | {rep.problems} |"
        )
        for field in ("calls", "accepted", "trades", "events", "problems"):
            setattr(total, field, getattr(total, field) + getattr(rep, field))
        if rep.problems:
            failures.append(f"{rep.regime}: {rep.problems} problems, first: {rep.first_problem}")
    print(
        f"| **all** | {total.calls:,} | {total.accepted:,} | {total.trades:,} | "
        f"{total.events:,} | {total.problems} |"
    )
    print(
        "\nChecked on every call: map and array results and events identical, accepted orders "
        "conserve quantity, neither book crossed. At the end: Python's FNV-1a over the events "
        "it received equals the C++ digest, validate() is clean, L3 books identical."
    )

    # 4. Real day, if present.
    day = args.lobster or find_lobster_day()
    if day is None:
        print("\n## Real LOBSTER day\n\nNo LOBSTER file in data/ (expected in CI); skipped.")
    else:
        print(f"\n## Real LOBSTER day from Python: {day.name}\n")
        print("| Replay | Events | Mismatches | Errors | Book events | Digest | Seconds |")
        print("|---|---:|---:|---:|---:|---|---:|")
        rows = replay_day(day)
        for r in rows:
            st = r["stats"]
            print(
                f"| {r['how']} | {st.events:,} | {st.mismatches} | {st.errors} | "
                f"{r['events']:,} | `{r['digest']:#018x}` | {r['seconds']:.2f} |"
            )
            if st.mismatches or st.errors:
                failures.append(f"{day.name} {r['how']}: {st.mismatches} mismatches, "
                                f"{st.errors} errors")
        if len({(r["digest"], r["events"]) for r in rows}) != 1:
            failures.append(f"{day.name}: replays disagree on the digest")
        known = KNOWN_DAYS.get(day.name)
        if known and (rows[0]["digest"], rows[0]["events"]) != known:
            failures.append(f"{day.name}: digest differs from lob_replay's ({known[0]:#x})")

    # Call cost, printed only.
    if not args.no_bench:
        print("\n## What a call costs from Python (not gated; machine-specific)\n")
        print("submit + cancel pairs at a standing level (2 events per call), 100,000 pairs, "
              "median [min–max] of 7 runs, ns per call:\n")
        print("| Book | Events | ns per call |")
        print("|---|---|---:|")
        configs = [
            ("recorded + digest (default), taken after each pair", {}),
            ("digest only", {"record_events": False}),
            ("neither", {"record_events": False, "digest_events": False}),
        ]
        for book_name, cls in (("ArrayOrderBook", ArrayOrderBook), ("OrderBook (map)", OrderBook)):
            for label, kw in configs:
                print(f"| {book_name} | {label} | {spread(call_cost(cls, **kw))} |")
        print(f"| empty Python function, same loop | – | {spread(empty_loop_cost())} |")

    print(f"\nGate: {'PASS' if not failures else 'FAIL'}")
    for f in failures:
        print(f"  - {f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
