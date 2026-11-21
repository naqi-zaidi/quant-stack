"""A simulated session written out as a LOBSTER message tape (Project 3, step 5).

The synthetic market runs every order through the C++ book, and the book reports what happened
as events. TapeRecorder turns those events into the rows a LOBSTER message file would have,
so the same replay code can run on the synthetic market and on Nasdaq's AAPL day:

  event 1  submit           an order that came to rest (its unfilled part, after any trades)
  event 2  partial cancel   a size cut that kept the order's place
  event 3  delete           a resting order cancelled, or moved to a new price (then resubmitted)
  event 4  execution        a resting order traded; direction is the resting order's side

Each row also carries the aggressor's agent code (market.agent_of of the incoming order), which
LOBSTER does not have. It lets the replay attribute a maker's fills to noise or informed flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orderbook import ExecReport, ExecType, Side, Trade

SUBMIT, CANCEL, DELETE, EXECUTE, EXECUTE_HIDDEN = 1, 2, 3, 4, 5
BUY, SELL = 1, -1  # LOBSTER direction
ID_SHIFT = 40  # as market.ID_SHIFT (not imported, to keep this module free of the simulator)


def _dir(side: Side) -> int:
    return BUY if side == Side.Buy else SELL


@dataclass
class Tape:
    """LOBSTER-format messages as numpy columns: ns, event, id, size, price (units), side
    (+1 buy / -1 sell, of the resting order) and aggressor (agent code, 0 if unknown)."""

    ns: np.ndarray
    event: np.ndarray
    id: np.ndarray
    size: np.ndarray
    price: np.ndarray
    side: np.ndarray
    aggressor: np.ndarray

    def __len__(self) -> int:
        return len(self.ns)


class TapeRecorder:
    """Fed every engine call's events (market._Simulator.call does this when asked)."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, int, int, int, int, int, int]] = []
        self.resting: dict[int, tuple[int, int, int]] = {}  # id -> (direction, price, qty)

    def record(self, events, book) -> None:
        new: list[ExecReport] = []
        for e in events:
            if isinstance(e, Trade):
                d, price, qty = self.resting[e.maker]
                taker_side = BUY if e.taker_side == Side.Buy else SELL
                self.rows.append((e.ts, EXECUTE, e.maker, e.qty, e.price, -taker_side,
                                  e.taker >> ID_SHIFT))
                if qty == e.qty:
                    del self.resting[e.maker]
                else:
                    self.resting[e.maker] = (d, price, qty - e.qty)
            elif isinstance(e, ExecReport):
                if e.type == ExecType.New:
                    new.append(e)
                elif e.type == ExecType.Cancelled and e.id in self.resting:
                    d, price, qty = self.resting.pop(e.id)
                    self.rows.append((e.ts, DELETE, e.id, qty, price, d, 0))
                elif e.type == ExecType.Replaced and e.id in self.resting:
                    d, price, qty = self.resting[e.id]
                    if e.price == price and e.leaves < qty:
                        self.rows.append((e.ts, CANCEL, e.id, qty - e.leaves, price, d, 0))
                        self.resting[e.id] = (d, price, e.leaves)
                    else:  # a new price: it loses its place, as a delete and a new order
                        self.rows.append((e.ts, DELETE, e.id, qty, price, d, 0))
                        del self.resting[e.id]
                        new.append(e)
        # Whatever came to rest during the call, after its trades (as LOBSTER orders them).
        for e in new:
            o = book.find(e.id)
            if o is not None and e.id not in self.resting:
                d = _dir(o.side)
                self.rows.append((e.ts, SUBMIT, e.id, o.qty, o.price, d, e.id >> ID_SHIFT))
                self.resting[e.id] = (d, o.price, o.qty)

    def tape(self) -> Tape:
        cols = list(zip(*self.rows, strict=True)) if self.rows else [()] * 7
        return Tape(*(np.array(c, dtype=np.int64) for c in cols))
