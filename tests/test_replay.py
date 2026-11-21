"""Tapes and the replay market maker with virtual orders (Project 3 step 5)."""

from __future__ import annotations

import numpy as np
import pytest

from mm_strategy.maker import FixedSpread, MarketMaker, Quotes
from mm_strategy.market import INFORMED, MarketParams, simulate
from mm_strategy.replay import (
    CROSSED,
    HIDDEN,
    QUEUE,
    THROUGH,
    ReplayMaker,
    TapeReplay,
    TapeSource,
    VirtualOrder,
)
from mm_strategy.tape import CANCEL, DELETE, EXECUTE, EXECUTE_HIDDEN, SUBMIT, Tape

SHORT = MarketParams(hours=0.05)
P = 1_000_000  # a price, units


def bid(ahead=300, cross=True):  # cross_fills on, to test it
    return VirtualOrder(side=1, price=P, qty=100, ahead=ahead, cross_fills=cross)


def test_queue_ahead_shrinks_only_for_orders_that_were_there_first() -> None:
    o = bid(ahead=300)
    assert o.on_message(SUBMIT, 7, 200, P, 1) == 0  # joins behind
    assert o.on_message(DELETE, 7, 200, P, 1) == 0 and o.ahead == 300  # behind: no change
    assert o.on_message(CANCEL, 1, 50, P, 1) == 0 and o.ahead == 250  # partial cancel ahead
    assert o.on_message(EXECUTE, 2, 200, P, 1) == 0 and o.ahead == 50  # executed ahead
    assert o.on_message(DELETE, 3, 50, P, 1) == 0 and o.ahead == 0
    assert o.on_message(DELETE, 3, 50, P - 100, 1) == 0  # another price: nothing
    assert o.on_message(EXECUTE, 9, 60, P, -1) == 0  # an ask executed: not our side


def test_fills_at_its_price_only_once_nothing_is_ahead() -> None:
    o = bid(ahead=100)
    o.on_message(SUBMIT, 7, 500, P, 1)  # behind us
    o.on_message(EXECUTE, 1, 100, P, 1)  # the order ahead trades
    assert o.ahead == 0 and o.qty == 100
    assert o.on_message(EXECUTE, 7, 60, P, 1) == 60 and o.kind == QUEUE  # then us, before #7
    assert o.on_message(EXECUTE, 7, 60, P, 1) == 40 and o.qty == 0


def test_trade_through_hidden_and_crossing_fills() -> None:
    o = bid(ahead=300)
    assert o.on_message(EXECUTE, 5, 30, P - 100, 1) == 30 and o.kind == THROUGH
    assert o.on_message(EXECUTE, 5, 30, P + 100, 1) == 0  # a better bid traded: not us
    assert o.on_message(EXECUTE_HIDDEN, 0, 10, P, 1) == 0  # hidden at our price, queue ahead
    assert o.on_message(EXECUTE_HIDDEN, 0, 10, P - 100, 1) == 10 and o.kind == HIDDEN
    assert o.on_message(SUBMIT, 8, 25, P + 100, -1) == 0  # an ask above us rests
    assert o.on_message(SUBMIT, 8, 25, P, -1) == 25 and o.kind == CROSSED  # an ask at our bid
    assert o.qty == 100 - 30 - 10 - 25
    no_cross = bid(cross=False)
    assert no_cross.on_message(SUBMIT, 8, 25, P - 100, -1) == 0
    ask = VirtualOrder(side=-1, price=P, qty=100, ahead=0, cross_fills=True)
    assert ask.on_message(EXECUTE, 4, 70, P + 100, -1) == 70  # buyer went through our ask
    assert ask.on_message(SUBMIT, 6, 50, P, 1) == 30  # a bid at our ask crosses


@pytest.fixture(scope="module")
def recorded():
    class Recorder:
        def __init__(self):
            self.views = []

        def quote(self, v):
            self.views.append((v.now, v.bids, v.asks))
            return Quotes(None, None)

    live = Recorder()
    s = simulate(SHORT, 11, participant=MarketMaker(live), record_tape=True)
    return s, live.views, Recorder


def test_tape_accounts_for_every_trade_and_order(recorded) -> None:
    s, _, _ = recorded
    t = s.tape
    assert isinstance(t, Tape) and len(t) > 1_000
    ex = t.event == EXECUTE
    assert t.size[ex].sum() == s.trades["qty"].sum()
    assert np.array_equal(t.price[ex], s.trades["price"])
    assert set(np.unique(t.event)) <= {SUBMIT, DELETE, EXECUTE}
    plain = simulate(SHORT, 11)
    assert plain.digest == s.digest  # recording changes nothing


def test_replay_shows_a_strategy_exactly_the_live_market(recorded) -> None:
    s, live_views, recorder = recorded
    rec = recorder()
    tr = TapeReplay(TapeSource(s.tape))
    tr.run([ReplayMaker(rec, 0, SHORT.duration_ns)], SHORT.duration_ns)
    assert rec.views == live_views
    assert tr.messages == len(s.tape)


def test_replay_maker_books_balance_and_fills_trace_to_the_tape(recorded) -> None:
    s, _, _ = recorded
    tr = TapeReplay(TapeSource(s.tape))
    rm = ReplayMaker(FixedSpread(1), 0, SHORT.duration_ns)
    tr.run([rm], SHORT.duration_ns)
    pnl = rm.pnl()
    f = rm.fill_arrays()
    assert pnl.identity_holds and len(f["ts"]) > 10
    assert rm.inventory == int((f["side"] * f["qty"]).sum())
    assert pnl.fees == -rm.fees.maker_rebate * pnl.volume
    assert set(np.unique(f["kind"])) <= {QUEUE, THROUGH}  # no hidden orders, no crossing
    assert np.isin(f["counterparty"], (2, INFORMED)).all()  # takers: noise market, informed
    assert (rm.orders[1], rm.orders[-1]) == (None, None)  # done at the close


def test_join_best_quotes_the_touch_within_its_limit() -> None:
    from mm_strategy.maker import JoinBest, MarketView

    v = MarketView(0, 1, 999_900, 1_000_100, 1_000_000.0, 0, 100)
    assert JoinBest().quote(v) == Quotes(999_900, 1_000_100, 100, 100)
    full = MarketView(0, 1, 999_900, 1_000_100, 1_000_000.0, 2_000, 100)
    assert JoinBest().quote(full).bid is None


EA, EB = 9_999_999_999, -9_999_999_999
B, A, MID = 5_853_300, 5_853_500, 5_853_400
# time, event, id, size, price, direction (the resting order's side)
DAY = [
    (34200.000, 1, 1, 100, B, 1),  # folded into the seed: an anonymous 100 at the bid
    (34200.010, 1, 2, 50, A, -1),
    # the maker wakes at 34200.100 and joins both queues: 100 ahead at the bid, 50 at the ask
    (34200.150, 1, 3, 40, B, 1),  # behind the maker's bid
    (34200.160, 4, 1, 100, B, 1),  # the order ahead trades: nothing left ahead
    (34200.170, 4, 3, 30, B, 1),  # an order behind trades: the maker would have gone first
    (34200.180, 5, 0, 20, MID, 1),  # hidden bid inside the spread: better than the maker's
    (34200.190, 5, 0, 25, MID, -1),  # hidden ask inside the spread: better than the maker's
    (34200.195, 4, 2, 50, A, -1),  # the order ahead of the maker's ask trades
    (34200.199, 1, 5, 10, A, -1),  # behind the maker's ask
    (34200.250, 4, 5, 10, A, -1),  # so this buyer reaches the maker first
]
BOOKS = [  # ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ask_s2, bid_p2, bid_s2 after each message
    (EA, 0, B, 100), (A, 50, B, 100), (A, 50, B, 140), (A, 50, B, 40), (A, 50, B, 10),
    (A, 50, B, 10), (A, 50, B, 10), (EA, 0, B, 10), (A, 10, B, 10), (EA, 0, B, 10),
]


def test_lobster_day_replay_fills_by_hand(tmp_path) -> None:
    from mm_strategy.maker import JoinBest
    from mm_strategy.replay import LobsterSource

    stem = tmp_path / "TEST_2012-06-21_34200000_57600000"
    with open(f"{stem}_message_2.csv", "w") as fh:
        fh.writelines(",".join(map(str, row)) + "\n" for row in DAY)
    with open(f"{stem}_orderbook_2.csv", "w") as fh:
        fh.writelines(",".join(map(str, (*row, EA, 0, EB, 0))) + "\n" for row in BOOKS)
    tr = TapeReplay(LobsterSource(f"{stem}_message_2.csv", levels=2), levels=2)
    t0 = 34_200 * 10**9
    rm = ReplayMaker(JoinBest(), t0, t0 + 300_000_000)
    tr.run([rm], t0 + 300_000_000)
    assert (tr.rp.stats.mismatches, tr.rp.stats.errors) == (0, 0)
    f = rm.fill_arrays()
    assert f["side"].tolist() == [1, -1] and f["qty"].tolist() == [30, 10]
    assert f["price"].tolist() == [B, A] and f["kind"].tolist() == [QUEUE, QUEUE]
    assert rm.inventory == 20 and rm.pnl().identity_holds


def test_a_one_sided_moment_is_skipped_when_pricing_a_fill() -> None:
    from mm_strategy.maker import JoinBest

    ms = 1_000_000
    rows = [  # ns, event, id, size, price, side, aggressor
        (1, SUBMIT, 1, 100, 999_900, 1, 1),
        (2, SUBMIT, 2, 100, 1_000_100, -1, 1),
        # the maker joins both queues at 100 ms, 100 shares ahead on each side
        (150 * ms, DELETE, 1, 100, 999_900, 1, 0),  # the bid side empties
        (160 * ms, SUBMIT, 3, 50, 1_000_100, -1, 1),  # behind the maker's ask
        (170 * ms, EXECUTE, 2, 100, 1_000_100, -1, 2),
        (180 * ms, EXECUTE, 3, 50, 1_000_100, -1, 2),  # the maker would have sold first
    ]
    tape = Tape(*(np.array(c, dtype=np.int64) for c in zip(*rows, strict=True)))
    tr = TapeReplay(TapeSource(tape))
    rm = ReplayMaker(JoinBest(), 0, 200 * ms)
    tr.run([rm], 200 * ms)
    pnl = rm.pnl()
    assert rm.fill_arrays()["qty"].tolist() == [50]
    # Priced against the last two-sided mid, 1,000,000: one tick of edge on 50 shares.
    assert pnl.spread == 50 * 100 and rm.mid2_end == 2_000_000 and pnl.identity_holds
