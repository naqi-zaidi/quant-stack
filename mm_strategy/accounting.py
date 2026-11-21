"""The market maker's books, recomputed from the engine's own trade record.

reconcile() rebuilds the maker's fills, inventory and cash from Session.trades: every trade the
C++ book printed with a MAKER order on either side. It checks them against what the maker
booked as fills arrived.

decompose() splits the session's profit, exactly, in integer price units:

    net PnL = spread capture + inventory PnL - fees
    spread capture = sum over fills of  side * qty * (mid before the fill - price)
    inventory PnL  = sum over fills of  side * qty * (mid at the close - mid before the fill)

Here side is +1 for a buy and -1 for a sell, and the mid is the book's, the maker's own quotes
included. Spread capture is what the maker earned against the mid at the moment it traded.
Inventory PnL is what the market then did to the position, which is where adverse selection
shows up. Mids are carried as bid + ask (twice the mid), so nothing is rounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mm_strategy.maker import MarketMaker
from mm_strategy.market import MAKER, NS, Session, agent_of

NO_BID, NO_ASK = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def maker_fills_from_engine(session: Session) -> dict[str, np.ndarray]:
    """The maker's fills as the engine printed them, in order."""
    tr = session.trades
    taker = agent_of(tr["taker"].astype(np.int64))
    maker = agent_of(tr["maker"].astype(np.int64))
    sel = (taker == MAKER) | (maker == MAKER)
    passive = maker[sel] == MAKER
    taker_side = tr["taker_side"][sel].astype(np.int64)
    side = np.where(passive, -taker_side, taker_side)
    other = np.where(passive, taker[sel], maker[sel])
    return {"ts": tr["ts"][sel].astype(np.int64), "side": side,
            "price": tr["price"][sel].astype(np.int64), "qty": tr["qty"][sel].astype(np.int64),
            "passive": passive, "counterparty": other.astype(np.int64)}


def reconcile(session: Session, mm: MarketMaker) -> list[str]:
    """Differences between the maker's own books and the engine's record (empty if none)."""
    eng, own = maker_fills_from_engine(session), mm.fill_arrays()
    problems = []
    if len(eng["ts"]) != len(own["ts"]):
        return [f"{len(own['ts'])} fills booked, {len(eng['ts'])} printed by the engine"]
    for k in eng:
        if not np.array_equal(eng[k], own[k]):
            problems.append(f"fill column {k} differs")
    inv = int((eng["side"] * eng["qty"]).sum())
    rebate, fee = mm.fees.maker_rebate, mm.fees.taker_fee
    fees = int(np.where(eng["passive"], -rebate, fee) @ eng["qty"])
    cash = int(-(eng["side"] * eng["price"] * eng["qty"]).sum()) - fees
    if inv != mm.inventory:
        problems.append(f"inventory {mm.inventory} booked vs {inv} from the engine")
    if cash != mm.cash:
        problems.append(f"cash {mm.cash} booked vs {cash} from the engine")
    if fees != mm.fees_paid:
        problems.append(f"fees {mm.fees_paid} booked vs {fees} from the engine")
    return problems


def mid2_series(session: Session) -> tuple[np.ndarray, np.ndarray]:
    """(ts, bid + ask) at every quote update with both sides quoted."""
    q = session.quotes
    two = (q["bid"] != NO_BID) & (q["ask"] != NO_ASK)
    return q["ts"][two].astype(np.int64), (q["bid"][two] + q["ask"][two]).astype(np.int64)


def mid2_before(session: Session, ts: np.ndarray) -> np.ndarray:
    """bid + ask in force just before each time (the last two-sided quote at an earlier ns)."""
    t, m2 = mid2_series(session)
    i = np.searchsorted(t, ts, side="left") - 1
    return m2[np.maximum(i, 0)]


@dataclass(frozen=True)
class PnL:
    """One session's profit in price units ($1 = 10,000)."""

    net: int
    spread: int
    inventory: int
    fees: int  # paid; negative = net rebate
    volume: int  # shares filled
    end_inventory: int
    identity_holds: bool

    def dollars(self, x: int) -> float:
        return x / 10_000


def split_pnl(side: np.ndarray, qty: np.ndarray, price: np.ndarray, mid2_before: np.ndarray,
              mid2_end: int, cash: int, inventory: int, fees: int) -> PnL:
    """The exact split, from fills (side +1 buy / -1 sell, qty, price), bid + ask just before
    each fill, bid + ask at the close, and the maker's own cash, inventory and fees."""
    sq = side * qty
    spread2 = int((sq * (mid2_before - 2 * price)).sum())
    inv2 = int((sq * (mid2_end - mid2_before)).sum())
    # Net PnL from the maker's own books, marked at the closing mid (all doubled: exact).
    net2 = 2 * cash + inventory * mid2_end
    holds = net2 == spread2 + inv2 - 2 * fees
    return PnL(net=net2 // 2, spread=spread2 // 2, inventory=inv2 // 2, fees=fees,
               volume=int(qty.sum()), end_inventory=inventory, identity_holds=holds)


def decompose(session: Session, mm: MarketMaker) -> PnL:
    f = maker_fills_from_engine(session)
    _, m2 = mid2_series(session)
    return split_pnl(f["side"], f["qty"], f["price"], mid2_before(session, f["ts"]), int(m2[-1]),
                     mm.cash, mm.inventory, mm.fees_paid)


def maker_markouts(session: Session, horizons_s: tuple[float, ...]) -> dict:
    """Volume-weighted markout of the maker's fills in ticks per share, per horizon, overall and
    by counterparty agent. Fills too late for a horizon, or whose horizon falls on a moment with
    one side of the book empty, are left out of that horizon."""
    f = maker_fills_from_engine(session)
    end = session.params.duration_ns
    out: dict = {}
    groups = {"all": np.ones(len(f["ts"]), bool)}
    for c in np.unique(f["counterparty"]):
        groups[int(c)] = f["counterparty"] == c
    for name, sel in groups.items():
        row = {}
        for h in horizons_s:
            t_h = f["ts"] + int(h * NS)
            m = f["side"] * (session.mid_at(t_h) - f["price"]) / session.params.tick
            ok = sel & (t_h <= end) & np.isfinite(m)  # skip moments with a one-sided book
            w = f["qty"][ok]
            row[h] = float((m[ok] * w).sum() / w.sum()) if w.sum() else float("nan")
        out[name] = row
    return out


def inventory_stats(session: Session, mm: MarketMaker) -> dict:
    """Max |inventory|, inventory half-life (AR(1) on 1-second samples) and max drawdown of the
    marked-to-mid PnL, from the maker's own samples."""
    s = np.array(mm.samples, dtype=np.int64)
    if len(s) < 3:
        return {"max_abs_inventory": 0, "half_life_s": float("nan"), "max_drawdown": 0.0}
    step = max(1, int(round(1000 / mm.requote_ms)))
    ts, inv, cash = s[::step, 0], s[::step, 1].astype(float), s[::step, 2].astype(float)
    pnl = cash + inv * session.mid_at(ts)
    pnl = pnl[np.isfinite(pnl)]
    dd = float((np.maximum.accumulate(pnl) - pnl).max()) if len(pnl) else 0.0
    x, y = inv[:-1], inv[1:]
    half_life = float("nan")
    if x.std() > 0:
        phi = float(np.polyfit(x, y, 1)[0])
        if 0 < phi < 1:
            half_life = float(np.log(0.5) / np.log(phi))
    return {"max_abs_inventory": int(np.abs(s[:, 1]).max()), "half_life_s": half_life,
            "max_drawdown": dd}
