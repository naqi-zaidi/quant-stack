"""Text forms of events, results and books: the format of lob/include/lob/text.hpp.

This is a separate Python implementation. orderbook.gate checks it against the C++-generated
golden transcripts in lob/tests/golden/, byte for byte.

    >>> from orderbook import ArrayOrderBook, Side
    >>> from orderbook.text import book_text, event_text
    >>> book = ArrayOrderBook()
    >>> _ = book.submit(1, Side.Sell, 101, 10)
    >>> print(book_text(book), end="")
      ask 101  10 in 1  [1:10]
      ----
      (no bids)
    >>> [event_text(e) for e in book.take_events()]
    ['report id=1 New Sell px=101 last=0 leaves=10 ts=0', 'level  Sell 101 qty=10 orders=1 ts=0']
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from orderbook._lob import ExecReport, LevelUpdate, Reason, Result, Side, Trade


def event_text(e: Trade | LevelUpdate | ExecReport) -> str:
    """One line per event, as text.hpp's to_text."""
    if isinstance(e, Trade):
        return (
            f"trade  taker={e.taker} maker={e.maker} {e.taker_side.name} {e.qty} @ {e.price} "
            f"ts={e.ts}"
        )
    if isinstance(e, LevelUpdate):
        return f"level  {e.side.name} {e.price} qty={e.qty} orders={e.count} ts={e.ts}"
    if isinstance(e, ExecReport):
        reason = "" if e.reason == Reason.None_ else f"({e.reason.name})"
        return (
            f"report id={e.id} {e.type.name}{reason} {e.side.name} px={e.price} "
            f"last={e.last_qty} leaves={e.leaves} ts={e.ts}"
        )
    raise TypeError(f"not an order book event: {e!r}")


def events_text(events: Iterable, indent: str = "  ") -> str:
    """Events one per line, each indented, as TextLog::take."""
    return "".join(f"{indent}{event_text(e)}\n" for e in events)


def result_text(r: Result) -> str:
    if not r.accepted:
        return f"rejected ({r.reason.name})"
    return f"accepted: filled {r.filled}, resting {r.resting}, cancelled {r.cancelled}"


def book_text(book) -> str:
    """Ladder: asks worst to best, a rule, bids best to worst. Each level shows its total,
    order count and FIFO queue as id:qty. Works with any book or book view."""

    def rows(side: Side) -> list[str]:
        tag = "ask" if side == Side.Sell else "bid"
        out = []
        for price, group in itertools.groupby(book.orders(side), key=lambda o: o.price):
            g = list(group)
            queue = " ".join(f"{o.id}:{o.qty}" for o in g)
            out.append(f"{tag} {price}  {sum(o.qty for o in g)} in {len(g)}  [{queue}]")
        return out

    asks, bids = rows(Side.Sell), rows(Side.Buy)
    text = "".join(f"  {r}\n" for r in reversed(asks)) or "  (no asks)\n"
    text += "  ----\n"
    text += "".join(f"  {r}\n" for r in bids) or "  (no bids)\n"
    return text
