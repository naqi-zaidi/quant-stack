"""Optimal execution (Phase 4 step 2): Almgren–Chriss schedules, TWAP and VWAP, on our own book.

The discrete Almgren–Chriss model, with the notation of their paper and no permanent impact:
sell X shares in N child orders, the k-th (k = 1..N) at t_{k-1} = t_0 + (k-1)τ, holding x_k shares
after it (x_0 = X, x_N = 0). A child order of n shares costs ε·n + η·n² ticks: half the spread
plus linear temporary impact. (η here is per share in the child order; the paper's η̃ is η·τ.)
The price diffuses with volatility σ ticks/√s while shares are held, so

    E = ε·X + η·Σ n_k²,      V = σ²·τ·Σ x_k²,

and minimising E + λ·V gives x_k = X·sinh(κ(T − t_k)) / sinh(κT), with cosh(κτ) = 1 + λσ²τ/(2η).
λ = 0 is TWAP; larger λ sells faster, paying more impact for less exposure to the price. The set
of (V, E) pairs over λ is the efficient frontier: no schedule has both lower cost and lower risk.

If the impact coefficient changes through the day (η_k), the first-order conditions are still a
tridiagonal linear system, solved by `optimal_holdings`. At λ = 0 its solution sells n_k ∝ 1/η_k:
when impact scales like 1/volume, the cost-minimising schedule is VWAP.

`Executor` sends a schedule into the synthetic market as market orders (agent EXECUTOR), and
`shortfall` measures implementation shortfall against the arrival mid (Perold, 1988), split into
three parts that add up exactly:
    arrival:  S0 − V0          the mid's mispricing when the order arrives (the same for every
                               schedule on a given day)
    timing:   Σ q (V0 − V_t)   the hidden fair value's move while shares wait: the risk
    impact:   Σ q (V_t − p)    what each fill gave up to the fair value at that moment: the cost
The split uses the fair value, which only a simulation knows. It makes the cost of a schedule
measurable to a fraction of a tick in a few sessions, though the price risk is ±40 ticks a share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from mm_strategy.market import EXECUTOR, NS, Session, agent_of
from orderbook import OrderType, Side


@dataclass(frozen=True)
class ImpactModel:
    """Cost of one child order of n shares: epsilon + eta·n ticks a share."""

    epsilon: float
    eta: float
    r2: float = float("nan")


# ---- Almgren–Chriss


def kappa(lam: float, sigma: float, tau: float, eta: float) -> float:
    """κ from cosh(κτ) = 1 + λσ²τ/(2η)."""
    return float(np.arccosh(1.0 + lam * sigma**2 * tau / (2.0 * eta)) / tau)


def lam_for_kappa(kappa_: float, sigma: float, tau: float, eta: float) -> float:
    """The risk aversion that gives a chosen κ: the inverse of `kappa`."""
    return float(2.0 * eta * (np.cosh(kappa_ * tau) - 1.0) / (sigma**2 * tau))


def ac_holdings(x0: float, n: int, tau: float, sigma: float, eta: float,
                lam: float) -> np.ndarray:
    """x_0..x_N of the Almgren–Chriss schedule; λ = 0 gives TWAP (a straight line)."""
    t = np.arange(n + 1) * tau
    big_t = n * tau
    if lam == 0:
        return x0 * (1.0 - t / big_t)
    k = kappa(lam, sigma, tau, eta)
    # sinh(k(T−t)) / sinh(kT), written to stay finite for large kT.
    a, b = k * (big_t - t), k * big_t
    return x0 * np.exp(a - b) * (-np.expm1(-2 * a)) / (-np.expm1(-2 * b))


def optimal_holdings(x0: float, eta: np.ndarray, sigma: float, tau: float,
                     lam: float) -> np.ndarray:
    """Minimise Σ η_k n_k² + λσ²τ Σ x_k² with a child-specific η_k (k = 1..N).

    Setting the derivative in x_j to zero gives
        −η_j x_{j−1} + (η_j + η_{j+1} + λσ²τ) x_j − η_{j+1} x_{j+1} = 0,   j = 1..N−1.
    """
    eta = np.asarray(eta, dtype=float)
    n = len(eta)
    if n == 1:
        return np.array([x0, 0.0])
    m = n - 1
    ab = np.zeros((3, m))
    ab[0, 1:] = -eta[1:m]
    ab[1, :] = eta[:m] + eta[1:] + lam * sigma**2 * tau
    ab[2, :-1] = -eta[1:m]
    rhs = np.zeros(m)
    rhs[0] = eta[0] * x0
    return np.r_[x0, solve_banded((1, 1), ab, rhs), 0.0]


def expected_cost(holdings: np.ndarray, model: ImpactModel) -> float:
    """E = ε·X + η·Σ n_k², in tick-shares."""
    n = -np.diff(np.asarray(holdings, dtype=float))
    return float(model.epsilon * n.sum() + model.eta * np.sum(n**2))


def variance(holdings: np.ndarray, sigma: float, tau: float) -> float:
    """V = σ²τ·Σ_{k=1}^{N−1} x_k², in tick-shares squared."""
    x = np.asarray(holdings, dtype=float)
    return float(sigma**2 * tau * np.sum(x[1:-1] ** 2))


def frontier_point(x0: float, n: int, tau: float, sigma: float, model: ImpactModel,
                   lam: float) -> tuple[float, float]:
    """Almgren & Chriss's closed forms for (E, V) on the efficient frontier (γ = 0)."""
    big_t, et = n * tau, model.eta * tau
    if lam == 0:
        return (model.epsilon * x0 + et * x0**2 / big_t,
                sigma**2 * x0**2 * big_t / 3 * (1 - 1 / n) * (1 - 1 / (2 * n)))
    k = kappa(lam, sigma, tau, model.eta)
    sh = np.sinh(k * big_t)
    e = model.epsilon * x0 + et * x0**2 * np.tanh(0.5 * k * tau) * (
        tau * np.sinh(2 * k * big_t) + 2 * big_t * np.sinh(k * tau)) / (2 * tau**2 * sh**2)
    v = 0.5 * sigma**2 * x0**2 * (tau * sh * np.cosh(k * (big_t - tau))
                                  - big_t * np.sinh(k * tau)) / (sh**2 * np.sinh(k * tau))
    return float(e), float(v)


def child_sizes(holdings: np.ndarray) -> np.ndarray:
    """Whole-share child orders that follow the holdings and add up to X exactly: round the
    cumulative amount sold, not each child."""
    x = np.asarray(holdings, dtype=float)
    sold = np.rint(x[0] - x).astype(np.int64)
    return np.diff(sold)


def twap(x0: int, n: int) -> np.ndarray:
    return ac_holdings(float(x0), n, 1.0, 1.0, 1.0, 0.0)


def vwap(x0: int, weights: np.ndarray) -> np.ndarray:
    """Holdings that sell in proportion to `weights` (expected market volume per child)."""
    w = np.asarray(weights, dtype=float)
    return x0 * (1.0 - np.r_[0.0, np.cumsum(w)] / w.sum())


# ---- execution on the synthetic market


@dataclass
class Executor:
    """A market participant that sends one child order per date as a market order.

    Sells by default. With `carry`, shares a market order can't fill (the bid side ran out) roll
    into the next child, and after the last child the rest is retried every `retry_s` seconds.
    `sides` (+1 buy, −1 sell, one per child) turns it into a probe for impact calibration.
    """

    start_s: float
    tau_s: float
    sizes: np.ndarray
    sides: np.ndarray | None = None
    carry: bool = True
    retry_s: float = 1.0
    fills: list = field(default_factory=list)  # (ts, price, qty, side)
    children: list = field(default_factory=list)  # (ts, wanted, filled, side, mid before)
    left: int = 0

    def on_start(self, sim) -> None:
        for k in range(len(self.sizes)):
            sim.schedule(int(round((self.start_s + k * self.tau_s) * NS)), sim.PARTICIPANT, k)

    def on_event(self, sim, k) -> None:
        last = len(self.sizes) - 1
        side = -1 if self.sides is None else int(self.sides[min(k, last)])
        want = (int(self.sizes[k]) if k <= last else 0) + (self.left if self.carry else 0)
        if want <= 0:
            return
        bid, ask = sim.book.best_bid(), sim.book.best_ask()
        mid = (bid + ask) / 2 if bid is not None and ask is not None else float("nan")
        r = sim.call(EXECUTOR, sim.book.submit, sim.new_id(EXECUTOR),
                     Side.Sell if side < 0 else Side.Buy, 0, want, OrderType.Market, sim.now)
        self.children.append((sim.now, want, r.filled, side, mid))
        self.left = want - r.filled if self.carry else 0
        if self.left > 0 and k >= last:
            sim.schedule(sim.now + int(self.retry_s * NS), sim.PARTICIPANT, k + 1)

    def on_fill(self, sim, trade) -> None:
        if agent_of(trade.taker) == EXECUTOR:
            side = 1 if trade.taker_side == Side.Buy else -1
            self.fills.append((trade.ts, trade.price, trade.qty, side))

    def fill_arrays(self) -> dict[str, np.ndarray]:
        cols = list(zip(*self.fills, strict=True)) if self.fills else [()] * 4
        return {k: np.array(c, dtype=np.int64)
                for k, c in zip(("ts", "price", "qty", "side"), cols, strict=True)}

    def timing_variance(self, sigma: float) -> float:
        """σ²·Σ x_j²·(t_{j+1} − t_j) over the children as executed: the variance of the timing
        term if the fair value moves σ ticks/√s (x_j = shares left after child j)."""
        ts = np.array([c[0] for c in self.children], dtype=float) / NS
        sold = np.cumsum([c[2] for c in self.children], dtype=float)
        left = sold[-1] - sold
        return float(sigma**2 * np.sum(left[:-1] ** 2 * np.diff(ts)))


@dataclass(frozen=True)
class Shortfall:
    """Implementation shortfall of a sale, in ticks a share (= bp on a $100 stock)."""

    filled: int
    total: float  # Σ q (S0 − p) / X
    arrival: float  # S0 − V0
    timing: float  # Σ q (V0 − V_t) / X
    impact: float  # Σ q (V_t − p) / X
    vs_mid: float  # Σ q (mid before the child − p) / X: the cost a trader can observe


def shortfall(session: Session, ex: Executor, arrival_s: float) -> Shortfall:
    tick = session.params.tick
    f = ex.fill_arrays()
    q = f["qty"].astype(float)
    x = q.sum()
    t_arr = int(round(arrival_s * NS))
    s0 = float(session.mid_at(t_arr - 1))
    v0 = float(session.fair_value_at(t_arr))
    vt = session.fair_value_at(f["ts"]).astype(float)
    p = f["price"].astype(float)
    mids = {c[0]: c[4] for c in ex.children}
    m = np.array([mids[t] for t in f["ts"]], dtype=float)
    return Shortfall(
        filled=int(x),
        total=float(np.sum(q * (s0 - p)) / x / tick),
        arrival=(s0 - v0) / tick,
        timing=float(np.sum(q * (v0 - vt)) / x / tick),
        impact=float(np.sum(q * (vt - p)) / x / tick),
        vs_mid=float(np.sum(q * (m - p)) / x / tick),
    )


def probe_costs(ex: Executor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per child order of a probe: (time s, shares filled, cost a share in price units against
    the mid just before it, positive when the order paid)."""
    f = ex.fill_arrays()
    out_t, out_n, out_c = [], [], []
    for ts, _want, filled, side, mid in ex.children:
        sel = f["ts"] == ts
        if filled == 0 or not np.isfinite(mid):
            continue
        q, p = f["qty"][sel].astype(float), f["price"][sel].astype(float)
        out_t.append(ts / NS)
        out_n.append(filled)
        out_c.append(side * float(np.sum(q * (p - mid))) / filled)
    return np.array(out_t), np.array(out_n, dtype=float), np.array(out_c)


def fit_impact(n: np.ndarray, cost_ticks: np.ndarray) -> ImpactModel:
    """OLS of cost a share (ticks) on [1, n]."""
    a = np.c_[np.ones_like(n), n]
    coef, *_ = np.linalg.lstsq(a, cost_ticks, rcond=None)
    resid = cost_ticks - a @ coef
    r2 = 1 - resid.var() / cost_ticks.var()
    return ImpactModel(float(coef[0]), float(coef[1]), float(r2))


def bucket_of(t_s: np.ndarray, start_s: float, end_s: float, buckets: int) -> np.ndarray:
    """Index of the equal-length time bucket each time falls in (clipped to the window)."""
    b = np.floor((np.asarray(t_s, dtype=float) - start_s) / (end_s - start_s) * buckets)
    return np.clip(b.astype(int), 0, buckets - 1)


@dataclass(frozen=True)
class BucketImpact:
    """Cost of a child order of n shares in bucket b: epsilon + eta[b]·n ticks a share."""

    epsilon: float
    eta: np.ndarray
    r2: float

    def per_child(self, t_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
        return self.eta[bucket_of(t_s, start_s, end_s, len(self.eta))]

    def expected_cost(self, holdings: np.ndarray, eta_k: np.ndarray) -> float:
        n = -np.diff(np.asarray(holdings, dtype=float))
        return float(self.epsilon * n.sum() + np.sum(eta_k * n**2))


def fit_bucket_impact(t_s: np.ndarray, n: np.ndarray, cost_ticks: np.ndarray, start_s: float,
                      end_s: float, buckets: int) -> BucketImpact:
    """OLS of cost a share on a common intercept and one slope in n per time bucket."""
    b = bucket_of(t_s, start_s, end_s, buckets)
    a = np.c_[np.ones_like(n), *(n * (b == j) for j in range(buckets))]
    coef, *_ = np.linalg.lstsq(a, cost_ticks, rcond=None)
    resid = cost_ticks - a @ coef
    return BucketImpact(float(coef[0]), coef[1:], float(1 - resid.var() / cost_ticks.var()))


def volume_profile(sessions: list[Session], start_s: float, end_s: float,
                   buckets: int) -> np.ndarray:
    """Average share of the window's market volume traded in each bucket."""
    edges = np.linspace(start_s, end_s, buckets + 1)
    shares = []
    for s in sessions:
        h, _ = np.histogram(s.trades["ts"] / NS, bins=edges, weights=s.trades["qty"])
        shares.append(h / h.sum())
    return np.mean(shares, axis=0)
