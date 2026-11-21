"""Markouts: how the mid moved after each fill, from that side's point of view.

For a fill at price p, on the side with sign s (+1 bought, -1 sold), the markout at horizon h is

    s * (mid(t + h) - p)

in ticks per share. It includes the half-spread earned or paid at the fill. A taker who pays the
spread starts negative; a passive order starts positive and loses that edge when informed flow
moves the price through it. Desks measure adverse selection this way. Fills too close to the end
of the session for a horizon are left out of that horizon.
"""

from __future__ import annotations

import numpy as np

from mm_strategy.market import NS, Session, agent_of


def fill_markouts(session: Session, horizons_s: tuple[float, ...]) -> dict[float, np.ndarray]:
    """Markout in ticks per share of every trade from the taker's side, per horizon (NaN where
    t + h runs past the session). The maker's markout is the negative."""
    tr = session.trades
    ts = tr["ts"].astype(np.int64)
    sign = tr["taker_side"].astype(float)
    price = tr["price"].astype(float)
    end = session.params.duration_ns
    out = {}
    for h in horizons_s:
        t_h = ts + int(h * NS)
        m = sign * (session.mid_at(t_h) - price) / session.params.tick
        out[h] = np.where(t_h <= end, m, np.nan)
    return out


def by_role(session: Session, horizons_s: tuple[float, ...]) -> dict[tuple[str, str], dict]:
    """Volume-weighted mean markout per (who, as what) and horizon, in ticks per share.

    Roles: ("<agent>", "taker"), and ("<maker agent>", "maker vs <taker agent>")."""
    from mm_strategy.market import AGENT_NAMES

    tr = session.trades
    qty = tr["qty"].astype(float)
    taker = agent_of(tr["taker"].astype(np.int64))
    maker = agent_of(tr["maker"].astype(np.int64))
    m = fill_markouts(session, horizons_s)
    groups: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    for a in np.unique(taker):
        groups[(AGENT_NAMES[int(a)], "taker")] = (taker == a, 1.0)
    for mk in np.unique(maker):
        for tk in np.unique(taker):
            sel = (maker == mk) & (taker == tk)
            if sel.any():
                groups[(AGENT_NAMES[int(mk)], f"maker vs {AGENT_NAMES[int(tk)]}")] = (sel, -1.0)
    out = {}
    for key, (sel, sgn) in groups.items():
        row = {}
        for h in horizons_s:
            ok = sel & np.isfinite(m[h])
            w = qty[ok]
            row[h] = float(sgn * (m[h][ok] * w).sum() / w.sum()) if w.sum() else float("nan")
        row["volume"] = float(qty[sel].sum())
        out[key] = row
    return out
