#pragma once

// Property-based testing for the order book, without an external framework.
//
//   generate_flow   seeded random order flow (five regimes), reproducible on any platform
//   check_call      the properties every call must satisfy, judged from the book state
//                   before and after the call and the call's outcome
//   EngineUnderTest the real engine plus its event stream, rebuilt and checked every call
//   run_properties  drive a flow through one book, stop at the first violation
//   run_differential drive a flow through the engine and the reference model in lockstep
//   shrink          cut a failing flow down to a minimal counterexample
//
// Ops are stored in a resolvable form (a modify says "size down by 3", not "set size 7"),
// so a flow stays meaningful after the shrinker deletes ops from it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lob/events.hpp"
#include "lob/order_book.hpp"
#include "reference_book.hpp"

namespace lob::testing_helpers {

// ---------------------------------------------------------------- randomness

// splitmix64. Unlike std::uniform_int_distribution, its output is identical on every
// compiler and standard library, so a seed means the same flow everywhere.
class Rng {
 public:
  explicit Rng(std::uint64_t seed) : s_(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  std::uint64_t below(std::uint64_t n) { return n == 0 ? 0 : next() % n; }
  std::int64_t between(std::int64_t lo, std::int64_t hi) {
    return lo + static_cast<std::int64_t>(below(static_cast<std::uint64_t>(hi - lo + 1)));
  }
  bool percent(int p) { return below(100) < static_cast<std::uint64_t>(p); }

 private:
  std::uint64_t s_;
};

// ---------------------------------------------------------------- order flow

struct Op {
  enum class Kind : std::uint8_t { Submit, Cancel, Modify };
  enum class Mod : std::uint8_t { SizeDown, SizeUp, Reprice, Same, Bad };
  Kind kind{Kind::Submit};
  Order order{};       // Submit
  OrderId target{};    // Cancel / Modify
  Mod mod{Mod::Same};  // Modify
  std::int64_t delta{};
};

// A concrete call, after a modify has been resolved against the current book.
struct Call {
  Op::Kind kind{Op::Kind::Submit};
  Order order{};  // Submit: the order. Cancel: id, ts. Modify: id, new price, new qty, ts.
};

struct Regime {
  std::string_view name;
  std::int64_t width;  // price levels each side of the inside that orders spread over
  int aggressive;      // % of new orders priced to cross
  int submit;          // % of ops that are new orders
  int cancel;          // % of ops that are cancels (the rest are modifies)
  Quantity max_size;
  int sweep;  // % of new orders that are large (100-400), sweeping several levels
};

inline constexpr std::array<Regime, 5> kRegimes{{
    {"tight spread", 4, 30, 55, 25, 30, 0},
    {"deep queues", 1, 15, 60, 20, 20, 0},
    {"wide book", 30, 10, 65, 20, 40, 0},
    {"cancel churn", 6, 20, 35, 40, 25, 0},
    {"sweeps", 10, 10, 55, 25, 30, 8},
}};

inline constexpr std::int64_t kMid = 1000;

inline std::vector<Op> generate_flow(const Regime& rg, std::uint64_t seed, std::size_t n) {
  Rng rng(seed);
  std::vector<Op> ops;
  ops.reserve(n);
  std::vector<OrderId> used;
  OrderId next_id = 1;
  auto recent = [&]() -> OrderId {  // a recently used id: usually still live
    const std::size_t k = std::min<std::size_t>(used.size(), 50);
    return used[used.size() - 1 - static_cast<std::size_t>(rng.below(k))];
  };
  auto new_order = [&](Timestamp ts) {
    const Side side = rng.percent(50) ? Side::Buy : Side::Sell;
    const auto t = rng.below(100);
    const OrderType type = t < 55   ? OrderType::Limit
                           : t < 70 ? OrderType::PostOnly
                           : t < 82 ? OrderType::IOC
                           : t < 92 ? OrderType::FOK
                                    : OrderType::Market;
    const std::int64_t off = rng.percent(rg.aggressive) ? -rng.between(0, rg.width)
                                                        : rng.between(1, rg.width + 1);
    const std::int64_t px = side == Side::Buy ? kMid - off : kMid + off;
    const Quantity qty = rng.percent(rg.sweep) ? rng.between(100, 400)
                                               : rng.between(1, rg.max_size);
    used.push_back(next_id);
    return Order{next_id++, side, type, Price{px}, qty, ts};
  };

  for (std::size_t i = 0; i < n; ++i) {
    const auto ts = static_cast<Timestamp>(i);
    const auto roll = static_cast<int>(rng.below(100));
    Op op;
    if (rng.percent(3) && !used.empty()) {  // malformed or stale input
      switch (rng.below(5)) {
        case 0:
          op.order = new_order(ts);
          op.order.qty = -rng.between(0, 5);
          break;
        case 1:
          op.order = new_order(ts);
          op.order.type = OrderType::Limit;
          op.order.price = Price{-rng.between(0, 5)};
          break;
        case 2: {
          const OrderId reuse = recent();  // a duplicate if still live, a legal reuse if not
          op.order = new_order(ts);
          op.order.id = reuse;
          used.pop_back();
          break;
        }
        case 3:
          op.kind = Op::Kind::Cancel;
          op.target = next_id + 1'000'000;  // never used
          break;
        default:
          op.kind = Op::Kind::Modify;
          op.target = recent();
          op.mod = Op::Mod::Bad;
          op.delta = rng.between(0, 1);
      }
    } else if (roll < rg.submit || used.empty()) {
      op.order = new_order(ts);
    } else if (roll < rg.submit + rg.cancel) {
      op.kind = Op::Kind::Cancel;
      op.target = recent();
    } else {
      op.kind = Op::Kind::Modify;
      op.target = recent();
      const auto m = rng.below(100);
      if (m < 35) {
        op.mod = Op::Mod::SizeDown;
        op.delta = rng.between(1, 10);
      } else if (m < 55) {
        op.mod = Op::Mod::SizeUp;
        op.delta = rng.between(1, 20);
      } else if (m < 95) {
        op.mod = Op::Mod::Reprice;
        op.delta = rng.between(1, rg.width + 3) * (rng.percent(50) ? 1 : -1);
      } else {
        op.mod = Op::Mod::Same;
      }
    }
    ops.push_back(op);
  }
  return ops;
}

[[nodiscard]] inline const OrderView* find_in(const BookState& st, OrderId id) {
  for (const auto* side : {&st.bids, &st.asks}) {
    for (const OrderView& o : *side) {
      if (o.id == id) return &o;
    }
  }
  return nullptr;
}

inline Call resolve(const Op& op, const BookState& st, Timestamp ts) {
  Call c{op.kind, op.order};
  if (op.kind == Op::Kind::Submit) {
    c.order.ts = ts;
    return c;
  }
  c.order = Order{op.target, Side::Buy, OrderType::Limit, Price{kMid}, 10, ts};
  const OrderView* o = find_in(st, op.target);
  if (op.kind == Op::Kind::Cancel || o == nullptr) return c;  // unknown id: rejected
  c.order.side = o->side;
  c.order.type = o->type;
  c.order.price = o->price;
  c.order.qty = o->qty;
  switch (op.mod) {
    case Op::Mod::SizeDown: c.order.qty = std::max<Quantity>(1, o->qty - op.delta); break;
    case Op::Mod::SizeUp: c.order.qty = o->qty + op.delta; break;
    case Op::Mod::Reprice:
      c.order.price = Price{std::max<std::int64_t>(1, o->price.ticks + op.delta)};
      break;
    case Op::Mod::Same: break;
    case Op::Mod::Bad:
      if (op.delta == 0) c.order.qty = 0;
      else c.order.price = Price{0};
      break;
  }
  return c;
}

template <typename Book>
Outcome apply(Book& b, const Call& c) {
  switch (c.kind) {
    case Op::Kind::Submit: return b.submit(c.order);
    case Op::Kind::Cancel: return b.cancel(c.order.id, c.order.ts);
    case Op::Kind::Modify: break;
  }
  return b.modify(c.order.id, c.order.price, c.order.qty, c.order.ts);
}

// ---------------------------------------------------------------- properties

struct Violation {
  std::string property;
  std::string detail;
};

namespace detail {

inline Quantity total(const BookState& st) {
  Quantity q = 0;
  for (const auto* side : {&st.bids, &st.asks}) {
    for (const OrderView& o : *side) q += o.qty;
  }
  return q;
}

inline std::unordered_set<OrderId> id_set(const BookState& st) {
  std::unordered_set<OrderId> ids;
  ids.reserve(st.bids.size() + st.asks.size());
  for (const auto* side : {&st.bids, &st.asks}) {
    for (const OrderView& o : *side) ids.insert(o.id);
  }
  return ids;
}

inline std::vector<OrderId> ids_except(const std::vector<OrderView>& v, OrderId skip) {
  std::vector<OrderId> out;
  for (const OrderView& o : v) {
    if (o.id != skip) out.push_back(o.id);
  }
  return out;
}

// Position of `id` within its price level's queue, or -1.
inline long queue_index(const std::vector<OrderView>& side, OrderId id) {
  const auto it = std::find_if(side.begin(), side.end(), [&](const auto& o) { return o.id == id; });
  if (it == side.end()) return -1;
  long k = 0;
  for (auto j = side.begin(); j != it; ++j) k += j->price == it->price ? 1 : 0;
  return k;
}

inline bool last_in_level(const std::vector<OrderView>& side, OrderId id) {
  const auto it = std::find_if(side.begin(), side.end(), [&](const auto& o) { return o.id == id; });
  return it != side.end() && (std::next(it) == side.end() || std::next(it)->price != it->price);
}

}  // namespace detail

// Every property a single call must satisfy. Returns the first one broken, in this order:
//   uncrossed            best bid < best ask whenever both sides have orders
//   accounting           filled + resting + cancelled adds up; a rejection changes nothing;
//                        the order's own resting state matches what the call reported
//   conservation         total resting quantity moves by exactly what the call reported,
//                        and printed trades add up to the filled quantity
//   price-time priority  the makers hit are exactly the front of the opposite side's
//                        queue as it stood before the call: best price first, oldest first
//   trade price          each trade prints at the maker's price, never through the limit
//   no early stop        an unfilled remainder means nothing crossable is left
//   order-type rules     market/IOC never rest, FOK fills in full, post-only never trades
//   queue position       orders the call didn't touch keep their relative order; a modify
//                        that only cuts size keeps its place, any other modify goes last
inline std::optional<Violation> check_call(const Call& c, const BookState& before,
                                           const Outcome& out, const BookState& after) {
  std::ostringstream d;
  auto fail = [&](std::string_view property) {
    return std::optional<Violation>{Violation{std::string(property), d.str()}};
  };
  const Result& r = out.result;
  const OrderId id = c.order.id;

  // uncrossed
  if (!after.bids.empty() && !after.asks.empty() && after.bids[0].price >= after.asks[0].price) {
    d << "bid " << after.bids[0].price.ticks << " >= ask " << after.asks[0].price.ticks;
    return fail("uncrossed");
  }

  // accounting
  if (!r.accepted) {
    if (r.filled != 0 || r.resting != 0 || r.cancelled != 0 || !out.trades.empty()) {
      d << "rejected call reported activity";
      return fail("accounting");
    }
    if (after != before) {
      d << "rejected call changed the book";
      return fail("accounting");
    }
    return std::nullopt;
  }
  const OrderView* prior = find_in(before, id);
  if (c.kind != Op::Kind::Submit && prior == nullptr) {
    d << "accepted a cancel/modify of an id that was not resting";
    return fail("accounting");
  }
  const Side side = c.kind == Op::Kind::Submit ? c.order.side : prior->side;
  const OrderType type = c.kind == Op::Kind::Submit ? c.order.type : prior->type;
  const Quantity sent = c.order.qty;
  const Quantity old_qty = prior != nullptr && c.kind != Op::Kind::Submit ? prior->qty : 0;
  switch (c.kind) {
    case Op::Kind::Submit:
      if (r.filled + r.resting + r.cancelled != sent) {
        d << "filled " << r.filled << " + resting " << r.resting << " + cancelled "
          << r.cancelled << " != sent " << sent;
        return fail("accounting");
      }
      break;
    case Op::Kind::Modify:
      if (r.filled + r.resting != sent || r.cancelled != 0) {
        d << "modify to " << sent << ": filled " << r.filled << " + resting " << r.resting;
        return fail("accounting");
      }
      break;
    case Op::Kind::Cancel:
      if (r.cancelled != old_qty || r.filled != 0 || r.resting != 0 || !out.trades.empty()) {
        d << "cancel of " << old_qty << " reported cancelled " << r.cancelled;
        return fail("accounting");
      }
      break;
  }
  const OrderView* now = find_in(after, id);
  if ((r.resting > 0) != (now != nullptr) || (now != nullptr && now->qty != r.resting)) {
    d << "reported resting " << r.resting << " but book holds "
      << (now != nullptr ? now->qty : 0);
    return fail("accounting");
  }
  const Price limit = c.order.price;  // the order's limit (new limit, for a modify)
  if (now != nullptr && (now->price != limit || now->side != side)) {
    d << "order rests at " << now->price.ticks << " instead of " << limit.ticks;
    return fail("accounting");
  }

  // conservation
  Quantity traded = 0;
  for (const Trade& t : out.trades) traded += t.qty;
  if (traded != r.filled) {
    d << "trades add up to " << traded << ", reported filled " << r.filled;
    return fail("conservation");
  }
  const Quantity moved = detail::total(after) - detail::total(before);
  const Quantity expect = r.resting - r.filled - old_qty;
  if (moved != expect) {
    d << "resting quantity moved by " << moved << ", expected " << expect;
    return fail("conservation");
  }

  // price-time priority
  const std::vector<OrderView>& opp = before.side(opposite(side));
  for (std::size_t i = 0; i < out.trades.size(); ++i) {
    const Trade& t = out.trades[i];
    if (i >= opp.size() || t.maker != opp[i].id || t.taker != id || t.taker_side != side) {
      d << "trade " << i + 1 << " hit order " << t.maker << ", but next in priority was "
        << (i < opp.size() ? static_cast<long long>(opp[i].id) : -1LL);
      return fail("price-time priority");
    }
    const bool last = i + 1 == out.trades.size();
    if (last ? t.qty > opp[i].qty : t.qty != opp[i].qty) {
      d << "trade " << i + 1 << " for " << t.qty << " against " << opp[i].qty
        << " resting: an earlier maker was left partly filled";
      return fail("price-time priority");
    }
  }

  // trade price
  for (const Trade& t : out.trades) {
    const OrderView* mk = find_in(before, t.maker);
    if (t.price != mk->price) {
      d << "trade printed at " << t.price.ticks << ", maker rests at " << mk->price.ticks;
      return fail("trade price");
    }
    if (type != OrderType::Market && !crosses(side, limit, t.price)) {
      d << "trade at " << t.price.ticks << " is through the limit " << limit.ticks;
      return fail("trade price");
    }
  }

  // no early stop (a cancel never matches, so this and the type rules skip it)
  const bool matched = c.kind != Op::Kind::Cancel;
  const std::vector<OrderView>& opp_after = after.side(opposite(side));
  if (matched && r.filled < sent && type != OrderType::PostOnly && !opp_after.empty() &&
      (type == OrderType::Market || crosses(side, limit, opp_after[0].price))) {
    d << "stopped with " << sent - r.filled << " unfilled while "
      << opp_after[0].price.ticks << " was still crossable";
    return fail("no early stop");
  }

  // order-type rules
  if (matched && (type == OrderType::Market || type == OrderType::IOC) && r.resting != 0) {
    d << to_string(type) << " order left " << r.resting << " resting";
    return fail("order-type rules");
  }
  if (matched && type == OrderType::FOK && r.filled != sent) {
    d << "fill-or-kill filled " << r.filled << " of " << sent;
    return fail("order-type rules");
  }
  if (matched && type == OrderType::PostOnly && r.filled != 0) {
    d << "post-only order traded " << r.filled;
    return fail("order-type rules");
  }

  // queue position
  const auto in_before = detail::id_set(before), in_after = detail::id_set(after);
  for (Side s : {Side::Buy, Side::Sell}) {
    std::vector<OrderId> pre = detail::ids_except(before.side(s), id);
    std::vector<OrderId> post = detail::ids_except(after.side(s), id);
    std::erase_if(pre, [&](OrderId x) { return !in_after.contains(x); });
    std::erase_if(post, [&](OrderId x) { return !in_before.contains(x); });
    if (pre != post) {
      d << to_string(s) << " side: orders the call did not touch changed relative order";
      return fail("queue position");
    }
  }
  if (c.kind == Op::Kind::Modify && now != nullptr) {
    const bool keeps = limit == prior->price && sent <= prior->qty;
    const auto& mine = after.side(side);
    if (keeps && detail::queue_index(mine, id) != detail::queue_index(before.side(side), id)) {
      d << "size-down from " << prior->qty << " to " << sent << " moved the order in its queue";
      return fail("queue position");
    }
    if (!keeps && !detail::last_in_level(mine, id)) {
      d << "modify to " << sent << " @ " << limit.ticks << " kept priority it should lose";
      return fail("queue position");
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------- the engine under test

// The real engine plus a subscriber that rebuilds per-order open quantity from exec reports
// and the L2 book from level updates, and checks both against the book after every call.
class EngineUnderTest {
 public:
  Outcome submit(const Order& o) {
    return run([&] { return book_.submit(o); });
  }
  Outcome cancel(OrderId id, Timestamp ts) {
    return run([&] { return book_.cancel(id, ts); });
  }
  Outcome modify(OrderId id, Price p, Quantity q, Timestamp ts) {
    return run([&] { return book_.modify(id, p, q, ts); });
  }
  [[nodiscard]] BookState state() const {
    return BookState{book_.orders(Side::Buy), book_.orders(Side::Sell)};
  }

  [[nodiscard]] std::optional<Violation> check_events(const BookState& after) {
    std::ostringstream d;
    for (const ExecReport& r : log_.reports) {
      if (r.type == ExecType::Rejected) continue;
      if (r.type == ExecType::Cancelled || r.leaves == 0) {
        open_.erase(r.id);
      } else {
        open_[r.id] = r.leaves;
      }
    }
    for (const LevelUpdate& u : log_.updates) {
      const auto key = std::make_pair(u.side == Side::Buy ? 0 : 1, u.price.ticks);
      if (u.qty == 0) {
        l2_.erase(key);
      } else {
        l2_[key] = std::make_pair(u.qty, u.count);
      }
    }
    decltype(l2_) truth;
    std::size_t resting = 0;
    for (Side s : {Side::Buy, Side::Sell}) {
      for (const OrderView& o : after.side(s)) {
        ++resting;
        auto& lv = truth[std::make_pair(s == Side::Buy ? 0 : 1, o.price.ticks)];
        lv.first += o.qty;
        ++lv.second;
        const auto it = open_.find(o.id);
        if (it == open_.end() || it->second != o.qty) {
          d << "exec reports say order " << o.id << " has "
            << (it == open_.end() ? 0 : it->second) << " open, book says " << o.qty;
          return Violation{"event stream", d.str()};
        }
      }
    }
    if (open_.size() != resting) {
      d << "exec reports leave " << open_.size() << " orders open, book holds " << resting;
      return Violation{"event stream", d.str()};
    }
    if (truth != l2_) {
      d << "L2 book rebuilt from level updates differs from the book";
      return Violation{"event stream", d.str()};
    }
    if (std::string err = book_.validate(); !err.empty()) {
      return Violation{"structure", err};
    }
    return std::nullopt;
  }

 private:
  template <typename F>
  Outcome run(F f) {
    log_.clear();
    const Result r = f();
    return Outcome{r, log_.trades};
  }

  EventLog log_;
  OrderBook book_{&log_, 64};  // small: the pool and index must grow during long flows
  std::unordered_map<OrderId, Quantity> open_;
  std::map<std::pair<int, std::int64_t>, std::pair<Quantity, std::uint32_t>> l2_;
};

// ---------------------------------------------------------------- runners

struct FlowResult {
  std::optional<Violation> violation;
  std::size_t step{0};  // index of the failing op
  std::size_t calls{0};
  std::size_t trades{0};
  std::size_t accepted{0};
};

template <typename Book>
FlowResult run_properties(const std::vector<Op>& ops, Book& book) {
  FlowResult fr;
  BookState before = book.state();
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const Call c = resolve(ops[i], before, static_cast<Timestamp>(i));
    const Outcome out = apply(book, c);
    BookState after = book.state();
    std::optional<Violation> v = check_call(c, before, out, after);
    if constexpr (requires { book.check_events(after); }) {
      if (!v) v = book.check_events(after);
    }
    ++fr.calls;
    fr.trades += out.trades.size();
    fr.accepted += out.result.accepted ? 1 : 0;
    if (v) {
      fr.violation = std::move(v);
      fr.step = i;
      return fr;
    }
    before = std::move(after);
  }
  return fr;
}

inline FlowResult run_properties(const std::vector<Op>& ops, Mutation m) {
  ReferenceBook ref(m);
  return run_properties(ops, ref);
}

// Engine and reference in lockstep: results, trades and full L3 state must be identical.
inline FlowResult run_differential(const std::vector<Op>& ops, Mutation m = Mutation::None) {
  FlowResult fr;
  ReferenceBook ref(m);
  EngineUnderTest eng;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const Call c = resolve(ops[i], ref.state(), static_cast<Timestamp>(i));
    const Outcome a = apply(ref, c);
    const Outcome b = apply(eng, c);
    ++fr.calls;
    fr.trades += b.trades.size();
    std::string what;
    if (a.result != b.result) what = "results differ";
    else if (a.trades != b.trades) what = "trades differ";
    else if (ref.state() != eng.state()) what = "resting orders differ";
    if (!what.empty()) {
      fr.violation = Violation{"differential", what};
      fr.step = i;
      return fr;
    }
  }
  return fr;
}

// Relabels order ids 1, 2, 3, ... in order of first appearance. A bijection on the ids in
// the flow, so behaviour is unchanged; it just makes counterexamples readable.
inline std::vector<Op> renumber(std::vector<Op> ops) {
  std::map<OrderId, OrderId> ids;
  auto relabel = [&](OrderId id) {
    return ids.try_emplace(id, static_cast<OrderId>(ids.size() + 1)).first->second;
  };
  for (Op& op : ops) {
    if (op.kind == Op::Kind::Submit) {
      op.order.id = relabel(op.order.id);
    } else {
      op.target = relabel(op.target);
    }
  }
  return ops;
}

// Delta debugging. `fails` must be true for `ops` on entry.
//   1. delete ever smaller chunks while the flow still fails (fast on long flows);
//   2. once the flow is short, delete single ops or pairs of ops until neither helps.
// Pairs matter: two calls can be needed only to set each other up (an order and the order
// that trades with it), so neither can go alone, but both can go together.
template <typename Fails>
std::vector<Op> shrink(std::vector<Op> ops, Fails fails) {
  auto without = [](const std::vector<Op>& v, std::size_t i, std::size_t len) {
    std::vector<Op> t(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i));
    t.insert(t.end(), v.begin() + static_cast<std::ptrdiff_t>(i + len), v.end());
    return t;
  };
  for (std::size_t chunk = std::max<std::size_t>(ops.size() / 2, 1);; chunk /= 2) {
    for (std::size_t start = 0; start + chunk <= ops.size();) {
      std::vector<Op> trial = without(ops, start, chunk);
      if (fails(trial)) {
        ops = std::move(trial);
      } else {
        start += chunk;
      }
    }
    if (chunk == 1) break;
  }
  for (bool progress = true; progress && ops.size() <= 64;) {
    progress = false;
    for (std::size_t i = 0; i < ops.size() && !progress; ++i) {
      if (std::vector<Op> t = without(ops, i, 1); fails(t)) {
        ops = std::move(t);
        progress = true;
      }
      for (std::size_t j = i + 1; j < ops.size() && !progress; ++j) {
        std::vector<Op> t = without(without(ops, j, 1), i, 1);
        if (fails(t)) {
          ops = std::move(t);
          progress = true;
        }
      }
    }
  }
  return renumber(std::move(ops));
}

// Replays a flow on `book` and prints each call and the first property it breaks.
template <typename Book>
std::string describe(const std::vector<Op>& ops, Book& ref) {
  std::ostringstream os;
  auto name = [](OrderId id) { return "#" + std::to_string(id); };
  BookState before = ref.state();
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const Call c = resolve(ops[i], before, static_cast<Timestamp>(i));
    const Outcome out = apply(ref, c);
    const BookState after = ref.state();
    os << "  " << i + 1 << ". ";
    switch (c.kind) {
      case Op::Kind::Submit:
        os << "submit " << name(c.order.id) << " " << to_string(c.order.side) << " "
           << to_string(c.order.type) << " " << c.order.qty;
        if (c.order.type != OrderType::Market) os << " @ " << c.order.price.ticks;
        break;
      case Op::Kind::Cancel: os << "cancel " << name(c.order.id); break;
      case Op::Kind::Modify:
        os << "modify " << name(c.order.id) << " to " << c.order.qty << " @ "
           << c.order.price.ticks;
        break;
    }
    const Result& r = out.result;
    if (!r.accepted) {
      os << "  -> rejected (" << to_string(r.reason) << ")";
    } else {
      os << "  -> filled " << r.filled << ", resting " << r.resting;
      if (r.cancelled > 0) os << ", cancelled " << r.cancelled;
      for (const Trade& t : out.trades) os << " [" << t.qty << " from " << name(t.maker) << "]";
    }
    if (auto v = check_call(c, before, out, after)) {
      os << "\n     violates " << v->property << ": " << v->detail;
    }
    os << "\n";
    before = after;
  }
  return os.str();
}

inline std::string describe(const std::vector<Op>& ops, Mutation m) {
  ReferenceBook ref(m);
  return describe(ops, ref);
}

// ---------------------------------------------------------------- seeded bugs

struct SeededBug {
  Mutation m;
  std::string_view key;   // identifier, for test names
  std::string_view name;  // what the bug does
};

inline constexpr std::array<SeededBug, 12> kSeededBugs{{
    {Mutation::LifoQueue, "LifoQueue", "newest order at a price fills first (LIFO)"},
    {Mutation::WorsePriceFirst, "WorsePriceFirst", "worst crossing price fills first"},
    {Mutation::TradeAtTakerPrice, "TradeAtTakerPrice", "trade prints at the taker's limit"},
    {Mutation::TradeThroughLimit, "TradeThroughLimit", "matching trades one tick through the limit"},
    {Mutation::StopAfterOneLevel, "StopAfterOneLevel", "matching stops after one price level"},
    {Mutation::FokFillsPartially, "FokFillsPartially", "fill-or-kill fills partially"},
    {Mutation::PostOnlyTakesAtEqualPrice, "PostOnlyTakesAtEqualPrice", "post-only trades at an equal price"},
    {Mutation::IocRestsRemainder, "IocRestsRemainder", "IOC remainder rests"},
    {Mutation::SizeDownLosesPriority, "SizeDownLosesPriority", "size-down loses queue priority"},
    {Mutation::SizeUpKeepsPriority, "SizeUpKeepsPriority", "size-up keeps queue priority"},
    {Mutation::RepriceIgnoresCross, "RepriceIgnoresCross", "re-price through the spread rests without trading"},
    {Mutation::RestsFullQuantity, "RestsFullQuantity", "full size rests after a partial fill"},
}};

}  // namespace lob::testing_helpers

namespace lob::testing_helpers {

// Searches seeded flows (cycling through the regimes) until the property suite catches the
// planted bug, then shrinks the failing prefix to a minimal flow that breaks the same
// property, and checks that the engine-vs-reference comparison catches it too.
struct Hunt {
  bool caught{false};
  Violation violation;
  std::string regime;
  std::uint64_t seed{0};
  std::size_t flows{0};           // flows tried, including the one that caught it
  std::size_t calls_to_catch{0};  // calls into that flow before the first violation
  std::vector<Op> shrunk;
  std::size_t differential_at{0};  // call at which engine and model first diverge (0 = never)
};

inline bool breaks(const std::vector<Op>& ops, Mutation m, const std::string& property) {
  const FlowResult r = run_properties(ops, m);
  return r.violation && r.violation->property == property;
}

inline Hunt hunt(Mutation m, std::size_t max_flows, std::size_t calls_per_flow) {
  Hunt h;
  for (std::size_t f = 0; f < max_flows; ++f) {
    const Regime& rg = kRegimes[f % kRegimes.size()];
    const std::uint64_t seed = f / kRegimes.size() + 1;
    std::vector<Op> ops = generate_flow(rg, seed, calls_per_flow);
    ++h.flows;
    const FlowResult fr = run_properties(ops, m);
    if (!fr.violation) continue;
    h.caught = true;
    h.violation = *fr.violation;
    h.regime = rg.name;
    h.seed = seed;
    h.calls_to_catch = fr.step + 1;
    const FlowResult diff = run_differential(ops, m);
    h.differential_at = diff.violation ? diff.step + 1 : 0;
    ops.resize(fr.step + 1);
    h.shrunk = shrink(ops, [&](const std::vector<Op>& o) { return breaks(o, m, h.violation.property); });
    break;
  }
  return h;
}

}  // namespace lob::testing_helpers
