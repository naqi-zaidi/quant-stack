#pragma once

// Microstructure analytics computed from the order book's event stream (Project 2, step 6).
//
// Every class here is a Listener. Its handlers (on_trade, on_level_update, on_report) update its
// state on each event; its eval functions compute values on demand. None of them reads the book:
// each rebuilds what it needs from events alone. So the same code serves a live book, a LOBSTER
// replay and Project 3's simulation. This is the handler/eval split of Jane Street's Flex Model.
//
//   SpreadDistribution  time-weighted distribution of the bid-ask spread, in ticks
//   DepthProfile        time-weighted mean resting size at 0..K-1 ticks from each side's best
//   OrderFlowImbalance  Cont, Kukanov and Stoikov's e_n at every change of the best quotes, a
//                       ring buffer of the last N with a running sum, and the cumulative total
//   QueuePositions      every resting order's place in its queue (orders and size ahead), with a
//                       history for the orders asked to be tracked
//   Fanout              hands one event stream to several listeners (a book has one slot)
//
// Time weighting. Each state is charged for the time it lasted: when an event arrives with a
// later timestamp than the last one, the state before it is charged for the gap. A state that
// only exists between events with the same timestamp (inside one message, or several messages
// stamped alike) is charged nothing. Timestamps must not decrease; an earlier one counts as no
// time passing. Eval functions taking `now` include the open interval up to `now` without
// changing state.
//
// reset() restarts the statistics but keeps the reconstructed book, so measurement can begin
// after seeding or after an opening period.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lob/events.hpp"
#include "lob/types.hpp"

namespace lob::analytics {

// ---------------------------------------------------------------- shared pieces

// The best bid and ask with their total sizes. An empty side has no price and size 0.
struct Quote {
  std::optional<Price> bid;
  Quantity bid_qty{0};
  std::optional<Price> ask;
  Quantity ask_qty{0};
  friend bool operator==(const Quote&, const Quote&) = default;
};

// Aggregate price levels rebuilt from LevelUpdates alone.
class L2Mirror {
 public:
  struct Level {
    Quantity qty{0};
    std::uint32_t count{0};
  };
  using Side_ = std::map<std::int64_t, Level>;  // ascending price; bids read from the back

  void apply(const LevelUpdate& u) {
    Side_& s = mut(u.side);
    if (u.qty == 0) {
      s.erase(u.price.ticks);
    } else {
      s[u.price.ticks] = Level{u.qty, u.count};
    }
  }

  [[nodiscard]] const Side_& side(lob::Side s) const { return s == lob::Side::Buy ? bids_ : asks_; }
  [[nodiscard]] Quote quote() const {
    Quote q;
    if (!bids_.empty()) {
      q.bid = Price{bids_.rbegin()->first};
      q.bid_qty = bids_.rbegin()->second.qty;
    }
    if (!asks_.empty()) {
      q.ask = Price{asks_.begin()->first};
      q.ask_qty = asks_.begin()->second.qty;
    }
    return q;
  }

  // Calls f(ticks_from_best, qty) for every level within `ticks` ticks of the side's best,
  // best first.
  template <typename F>
  void near_best(lob::Side s, std::int64_t tick, std::size_t ticks, F&& f) const {
    const auto k_of = [tick](std::int64_t best, std::int64_t px) {
      return static_cast<std::size_t>((best > px ? best - px : px - best) / tick);
    };
    if (s == lob::Side::Buy) {
      if (bids_.empty()) return;
      const std::int64_t best = bids_.rbegin()->first;
      for (auto it = bids_.rbegin(); it != bids_.rend(); ++it) {
        const std::size_t k = k_of(best, it->first);
        if (k >= ticks) break;
        f(k, it->second.qty);
      }
    } else {
      if (asks_.empty()) return;
      const std::int64_t best = asks_.begin()->first;
      for (auto it = asks_.begin(); it != asks_.end(); ++it) {
        const std::size_t k = k_of(best, it->first);
        if (k >= ticks) break;
        f(k, it->second.qty);
      }
    }
  }

 private:
  Side_& mut(lob::Side s) { return s == lob::Side::Buy ? bids_ : asks_; }
  Side_ bids_;
  Side_ asks_;
};

namespace detail {

// a + b * c in unsigned 64 bits, saturating (and setting `overflow`) rather than wrapping. A
// listener runs inside a book call, so it must not throw.
inline std::uint64_t add_mul(std::uint64_t a, std::uint64_t b, std::uint64_t c, bool& overflow) {
  constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
  if (b != 0 && c > kMax / b) {
    overflow = true;
    return kMax;
  }
  const std::uint64_t bc = b * c;
  if (a > kMax - bc) {
    overflow = true;
    return kMax;
  }
  return a + bc;
}

}  // namespace detail

// Base for time-weighted statistics: charges the current state for each gap between event
// timestamps, before the later event is applied.
class TimeWeighted : public Listener {
 public:
  void on_trade(const Trade& t) override { advance(t.ts); }
  void on_report(const ExecReport& r) override { advance(r.ts); }
  void on_level_update(const LevelUpdate& u) override {
    advance(u.ts);
    book_.apply(u);
  }

  // Restarts the statistics at `now` (or at the last event's time if that is later), keeping
  // the reconstructed book.
  void reset(Timestamp now) {
    clear();
    started_ = true;
    now_ = std::max(now, now_);
    start_ = now_;
  }

  [[nodiscard]] Quote quote() const { return book_.quote(); }
  [[nodiscard]] const L2Mirror& book() const noexcept { return book_; }
  // Time covered so far, up to `now`.
  [[nodiscard]] Timestamp elapsed(Timestamp now) const {
    return started_ ? std::max(now, now_) - start_ : 0;
  }
  // True if a sum hit 2^64 - 1 and stopped there (never for a trading day of real data).
  [[nodiscard]] bool overflowed() const noexcept { return overflow_; }

 protected:
  // Time between the last event and `now` (zero if `now` is not later).
  [[nodiscard]] Timestamp open_interval(Timestamp now) const {
    return started_ && now > now_ ? now - now_ : 0;
  }
  virtual void charge(Timestamp dt) = 0;  // the current state lasted dt
  virtual void clear() = 0;               // forget the statistics

  L2Mirror book_;
  bool overflow_{false};

 private:
  void advance(Timestamp ts) {
    if (!started_) {
      started_ = true;
      now_ = start_ = ts;
    } else if (ts > now_) {
      charge(ts - now_);
      now_ = ts;
    }
  }
  Timestamp now_{0};
  Timestamp start_{0};
  bool started_{false};
};

// ---------------------------------------------------------------- spread

class SpreadDistribution final : public TimeWeighted {
 public:
  explicit SpreadDistribution(std::int64_t tick_size = 1) : tick_(tick_size > 0 ? tick_size : 1) {}

  // Current spread in ticks, if both sides are quoted.
  [[nodiscard]] std::optional<std::int64_t> spread() const {
    const Quote q = book_.quote();
    if (!q.bid || !q.ask) return std::nullopt;
    return (q.ask->ticks - q.bid->ticks) / tick_;
  }
  // (spread in ticks, nanoseconds spent there), ascending by spread, up to `now`.
  [[nodiscard]] std::vector<std::pair<std::int64_t, std::uint64_t>> histogram(Timestamp now) const {
    std::map<std::int64_t, std::uint64_t> h = ns_;
    const Timestamp extra = open_interval(now);
    if (extra > 0) {
      if (const auto s = spread()) h[*s] += extra;
    }
    return {h.begin(), h.end()};
  }
  // Time with both sides quoted, and with at least one side empty.
  [[nodiscard]] std::uint64_t two_sided_ns(Timestamp now) const {
    return two_sided_ + (spread() ? open_interval(now) : 0);
  }
  [[nodiscard]] std::uint64_t one_sided_ns(Timestamp now) const {
    return one_sided_ + (spread() ? 0 : open_interval(now));
  }
  // Time-weighted mean spread in ticks over two-sided time (NaN before any).
  [[nodiscard]] double mean(Timestamp now) const {
    double num = 0;
    double den = 0;
    for (const auto& [s, ns] : histogram(now)) {
      num += static_cast<double>(s) * static_cast<double>(ns);
      den += static_cast<double>(ns);
    }
    return den > 0 ? num / den : std::numeric_limits<double>::quiet_NaN();
  }
  [[nodiscard]] std::int64_t tick_size() const noexcept { return tick_; }

 private:
  void charge(Timestamp dt) override {
    if (const auto s = spread()) {
      ns_[*s] += dt;
      two_sided_ += dt;
    } else {
      one_sided_ += dt;
    }
  }
  void clear() override {
    ns_.clear();
    two_sided_ = one_sided_ = 0;
  }

  std::int64_t tick_;
  std::map<std::int64_t, std::uint64_t> ns_;
  std::uint64_t two_sided_{0};
  std::uint64_t one_sided_{0};
};

// ---------------------------------------------------------------- depth profile

class DepthProfile final : public TimeWeighted {
 public:
  // Depth at 0, 1, ..., ticks-1 ticks from each side's best price.
  explicit DepthProfile(std::int64_t tick_size = 1, std::size_t ticks = 10)
      : tick_(tick_size > 0 ? tick_size : 1),
        ticks_(ticks > 0 ? ticks : 1),
        sums_{std::vector<std::uint64_t>(ticks_), std::vector<std::uint64_t>(ticks_)} {}

  // Resting size at each distance from the best, now (zeros where no level sits).
  [[nodiscard]] std::vector<Quantity> current(Side s) const {
    std::vector<Quantity> d(ticks_, 0);
    book_.near_best(s, tick_, ticks_, [&](std::size_t k, Quantity q) { d[k] = q; });
    return d;
  }
  // Sum over time of size x nanoseconds at each distance, up to `now`.
  [[nodiscard]] std::vector<std::uint64_t> sums(Side s, Timestamp now) const {
    std::vector<std::uint64_t> out = sums_[idx(s)];
    if (const Timestamp extra = open_interval(now); extra > 0) {
      bool ignored = false;
      book_.near_best(s, tick_, ticks_, [&](std::size_t k, Quantity q) {
        out[k] = detail::add_mul(out[k], static_cast<std::uint64_t>(q), extra, ignored);
      });
    }
    return out;
  }
  // Time during which the side had at least one level, up to `now`.
  [[nodiscard]] std::uint64_t observed_ns(Side s, Timestamp now) const {
    return observed_[idx(s)] + (book_.side(s).empty() ? 0 : open_interval(now));
  }
  // Time-weighted mean size at each distance, over the time the side was quoted.
  [[nodiscard]] std::vector<double> mean(Side s, Timestamp now) const {
    const auto sum = sums(s, now);
    const auto t = static_cast<double>(observed_ns(s, now));
    std::vector<double> m(ticks_, std::numeric_limits<double>::quiet_NaN());
    if (t > 0) {
      for (std::size_t k = 0; k < ticks_; ++k) m[k] = static_cast<double>(sum[k]) / t;
    }
    return m;
  }
  [[nodiscard]] std::size_t ticks() const noexcept { return ticks_; }
  [[nodiscard]] std::int64_t tick_size() const noexcept { return tick_; }

 private:
  static constexpr std::size_t idx(Side s) noexcept { return s == Side::Buy ? 0 : 1; }
  void charge(Timestamp dt) override {
    for (const Side s : {Side::Buy, Side::Sell}) {
      if (book_.side(s).empty()) continue;
      observed_[idx(s)] += dt;
      auto& sum = sums_[idx(s)];
      book_.near_best(s, tick_, ticks_, [&](std::size_t k, Quantity q) {
        sum[k] = detail::add_mul(sum[k], static_cast<std::uint64_t>(q), dt, overflow_);
      });
    }
  }
  void clear() override {
    for (auto& v : sums_) std::fill(v.begin(), v.end(), 0);
    observed_[0] = observed_[1] = 0;
  }

  std::int64_t tick_;
  std::size_t ticks_;
  std::vector<std::uint64_t> sums_[2];
  std::uint64_t observed_[2]{0, 0};
};

// ---------------------------------------------------------------- order-flow imbalance

// One change of the best quotes, and its contribution e_n.
struct QuoteUpdate {
  Timestamp ts{};
  Quote quote;
  std::int64_t e{0};
};

// Cont, Kukanov and Stoikov (2014), "The Price Impact of Order Book Events": for consecutive
// best quotes (P^B, q^B, P^A, q^A),
//   e_n = 1{P^B_n >= P^B_n-1} q^B_n - 1{P^B_n <= P^B_n-1} q^B_n-1
//       - 1{P^A_n <= P^A_n-1} q^A_n + 1{P^A_n >= P^A_n-1} q^A_n-1
// An empty bid counts as price -infinity, an empty ask as +infinity, both with size 0.
[[nodiscard]] inline std::int64_t ofi_contribution(const Quote& prev, const Quote& cur) {
  constexpr std::int64_t kLow = std::numeric_limits<std::int64_t>::min();
  constexpr std::int64_t kHigh = std::numeric_limits<std::int64_t>::max();
  const std::int64_t pb0 = prev.bid ? prev.bid->ticks : kLow;
  const std::int64_t pb1 = cur.bid ? cur.bid->ticks : kLow;
  const std::int64_t pa0 = prev.ask ? prev.ask->ticks : kHigh;
  const std::int64_t pa1 = cur.ask ? cur.ask->ticks : kHigh;
  std::int64_t e = 0;
  if (pb1 >= pb0) e += cur.bid_qty;
  if (pb1 <= pb0) e -= prev.bid_qty;
  if (pa1 <= pa0) e -= cur.ask_qty;
  if (pa1 >= pa0) e += prev.ask_qty;
  return e;
}

class OrderFlowImbalance final : public Listener {
 public:
  // `window`: how many of the latest quote updates window_sum() covers. `record`: keep every
  // QuoteUpdate (for regressions and figures).
  explicit OrderFlowImbalance(std::size_t window = 100, bool record = false)
      : ring_(window > 0 ? window : 1, 0), record_(record) {}

  void on_level_update(const LevelUpdate& u) override {
    book_.apply(u);
    const Quote q = book_.quote();
    if (q == prev_) return;  // a deeper level changed: not a quote update
    const std::int64_t e = ofi_contribution(prev_, q);
    prev_ = q;
    window_sum_ += e - ring_[head_];
    ring_[head_] = e;
    head_ = (head_ + 1) % ring_.size();
    if (filled_ < ring_.size()) ++filled_;
    total_ += e;
    last_ = e;
    ++updates_;
    if (record_) records_.push_back(QuoteUpdate{u.ts, q, e});
  }

  // Restarts the window, total, count and record; keeps the current quote.
  void reset() {
    std::fill(ring_.begin(), ring_.end(), 0);
    head_ = filled_ = 0;
    window_sum_ = total_ = last_ = 0;
    updates_ = 0;
    records_.clear();
  }

  [[nodiscard]] std::int64_t last() const noexcept { return last_; }
  [[nodiscard]] std::int64_t window_sum() const noexcept { return window_sum_; }
  [[nodiscard]] std::size_t window() const noexcept { return ring_.size(); }
  [[nodiscard]] std::size_t window_count() const noexcept { return filled_; }  // <= window
  [[nodiscard]] std::int64_t total() const noexcept { return total_; }
  [[nodiscard]] std::size_t updates() const noexcept { return updates_; }
  [[nodiscard]] Quote quote() const { return prev_; }
  // Queue imbalance at the best quotes, (q^B - q^A) / (q^B + q^A), if both sides are quoted.
  [[nodiscard]] std::optional<double> imbalance() const {
    if (!prev_.bid || !prev_.ask) return std::nullopt;
    return static_cast<double>(prev_.bid_qty - prev_.ask_qty) /
           static_cast<double>(prev_.bid_qty + prev_.ask_qty);
  }
  [[nodiscard]] bool recording() const noexcept { return record_; }
  [[nodiscard]] const std::vector<QuoteUpdate>& records() const noexcept { return records_; }

 private:
  L2Mirror book_;
  Quote prev_;
  std::vector<std::int64_t> ring_;
  std::size_t head_{0};
  std::size_t filled_{0};
  std::int64_t window_sum_{0};
  std::int64_t total_{0};
  std::int64_t last_{0};
  std::size_t updates_{0};
  bool record_;
  std::vector<QuoteUpdate> records_;
};

// ---------------------------------------------------------------- queue positions

struct QueuePosition {
  std::size_t orders{0};  // orders ahead in the same queue
  Quantity qty{0};        // their open size
  friend bool operator==(const QueuePosition&, const QueuePosition&) = default;
};

enum class QueueStatus : std::uint8_t { Resting, Filled, Cancelled };

// A tracked order's place, recorded whenever it changed (at the end of the timestamp), and once
// more when the order leaves the book.
struct QueueSample {
  Timestamp ts{};
  OrderId id{};
  QueuePosition ahead;
  Quantity leaves{0};
  QueueStatus status{QueueStatus::Resting};
};

// A resting order as the mirror sees it, for comparison with the book's L3 view.
struct MirroredOrder {
  OrderId id{};
  Price price{};
  Quantity leaves{0};
  friend bool operator==(const MirroredOrder&, const MirroredOrder&) = default;
};

// Every order's queue, rebuilt from execution reports alone, following the engine's priority
// rules: an order joins the back of its level when it arrives (New) or when it is replaced at a
// new price or a larger size; a size cut at the same price keeps its place; fills reduce it and
// remove it at zero; a cancel removes it. Rejections change nothing.
//
// An aggressive order is placed in its level when its New report arrives, before it trades, and
// leaves (or rests) by the end of the same call: such passing states never last any time, and
// tracked histories are recorded only when time moves on.
class QueuePositions final : public Listener {
 public:
  void on_trade(const Trade& t) override { advance(t.ts); }
  void on_level_update(const LevelUpdate& u) override { advance(u.ts); }
  void on_report(const ExecReport& r) override {
    advance(r.ts);
    switch (r.type) {
      case ExecType::New:
        if (orders_.count(r.id) != 0) remove(r.id, QueueStatus::Cancelled, r.ts);
        add(r.id, r.side, r.price.ticks, r.leaves);
        break;
      case ExecType::Replaced: {
        auto it = orders_.find(r.id);
        if (it == orders_.end()) break;
        Node& n = it->second;
        if (r.price.ticks == n.price && r.leaves < n.leaves) {
          n.leaves = r.leaves;  // size cut in place: keeps its place
          touch(n);
        } else {
          const Node was = n;  // copied: unlink() erases the node
          unlink(it);
          add(r.id, was.side, r.price.ticks, r.leaves, was.tracked);
          Node& moved = orders_.at(r.id);  // a new place is recorded only if it differs
          moved.last_ahead = was.last_ahead;
          moved.last_leaves = was.last_leaves;
        }
        break;
      }
      case ExecType::Fill: {
        auto it = orders_.find(r.id);
        if (it == orders_.end()) break;
        if (r.leaves <= 0) {
          remove(r.id, QueueStatus::Filled, r.ts);
        } else {
          it->second.leaves = r.leaves;
          touch(it->second);
        }
        break;
      }
      case ExecType::Cancelled:
        remove(r.id, QueueStatus::Cancelled, r.ts);
        break;
      case ExecType::Rejected:
        break;
    }
  }

  // Orders and size ahead of `id` in its queue, if it is resting.
  [[nodiscard]] std::optional<QueuePosition> ahead(OrderId id) const {
    const auto it = orders_.find(id);
    if (it == orders_.end()) return std::nullopt;
    const Node& n = it->second;
    const Queue& q = levels(n.side).at(n.price);
    QueuePosition pos;
    for (OrderId other : q.fifo) {
      if (other == id) break;
      ++pos.orders;
      pos.qty += orders_.at(other).leaves;
    }
    return pos;
  }

  // Records `id`'s place whenever it changes, until it leaves the book. The order may arrive
  // later; it is tracked from its New report.
  void track(OrderId id) {
    if (auto it = orders_.find(id); it != orders_.end()) {
      if (!it->second.tracked) {
        it->second.tracked = true;
        ++levels(it->second.side)[it->second.price].tracked;
        touch(it->second);
      }
    } else {
      wanted_.insert(id);
    }
  }

  // Records pending changes for tracked orders now (call at the end of a run).
  void flush() { record_dirty(now_); }
  [[nodiscard]] const std::vector<QueueSample>& history() const noexcept { return history_; }
  void reset() { history_.clear(); }

  [[nodiscard]] std::size_t order_count() const noexcept { return orders_.size(); }
  // Every resting order on one side, in priority order: best price first, then FIFO.
  [[nodiscard]] std::vector<MirroredOrder> orders(Side s) const {
    std::vector<MirroredOrder> out;
    const auto emit_level = [&](std::int64_t px, const Queue& q) {
      for (OrderId id : q.fifo) out.push_back({id, Price{px}, orders_.at(id).leaves});
    };
    if (s == Side::Buy) {
      for (auto it = bids_.rbegin(); it != bids_.rend(); ++it) emit_level(it->first, it->second);
    } else {
      for (const auto& [px, q] : asks_) emit_level(px, q);
    }
    return out;
  }

 private:
  struct Queue {
    std::list<OrderId> fifo;
    std::size_t tracked{0};
    bool dirty{false};
  };
  struct Node {
    Side side{Side::Buy};
    std::int64_t price{0};
    Quantity leaves{0};
    std::list<OrderId>::iterator pos;
    bool tracked{false};
    QueuePosition last_ahead{};
    Quantity last_leaves{-1};  // last recorded leaves; -1 = nothing recorded yet
  };
  using Levels = std::map<std::int64_t, Queue>;

  Levels& levels(Side s) { return s == Side::Buy ? bids_ : asks_; }
  [[nodiscard]] const Levels& levels(Side s) const { return s == Side::Buy ? bids_ : asks_; }

  void advance(Timestamp ts) {
    if (ts > now_) {
      record_dirty(now_);
      now_ = ts;
    }
  }

  void add(OrderId id, Side s, std::int64_t price, Quantity leaves, bool tracked = false) {
    if (wanted_.erase(id) != 0) tracked = true;
    Queue& q = levels(s)[price];
    q.fifo.push_back(id);
    Node n{s, price, leaves, std::prev(q.fifo.end()), tracked, {}, -1};
    if (tracked) ++q.tracked;
    const auto [it, ok] = orders_.insert_or_assign(id, n);
    touch(it->second);
  }

  // Takes the order out of its queue (and drops an emptied queue), keeping nothing.
  void unlink(std::unordered_map<OrderId, Node>::iterator it) {
    Node& n = it->second;
    Levels& lv = levels(n.side);
    auto qit = lv.find(n.price);
    Queue& q = qit->second;
    q.fifo.erase(n.pos);
    if (n.tracked) --q.tracked;
    if (q.tracked > 0 && !q.dirty) {  // the orders behind it moved up
      q.dirty = true;
      dirty_.emplace_back(n.side, n.price);
    }
    if (q.fifo.empty()) lv.erase(qit);
    orders_.erase(it);
  }

  void remove(OrderId id, QueueStatus why, Timestamp ts) {
    auto it = orders_.find(id);
    if (it == orders_.end()) return;
    if (it->second.tracked) history_.push_back({ts, id, {}, 0, why});
    unlink(it);
  }

  // The order's queue changed: if it holds tracked orders, re-check them when time moves on.
  void touch(const Node& n) {
    Queue& q = levels(n.side)[n.price];
    if (q.tracked > 0 && !q.dirty) {
      q.dirty = true;
      dirty_.emplace_back(n.side, n.price);
    }
  }

  void record_dirty(Timestamp ts) {
    for (const auto& [s, px] : dirty_) {
      Levels& lv = levels(s);
      auto qit = lv.find(px);
      if (qit == lv.end() || !qit->second.dirty) continue;
      Queue& q = qit->second;
      q.dirty = false;
      QueuePosition pos;
      for (OrderId id : q.fifo) {
        Node& n = orders_.at(id);
        if (n.tracked && (n.last_leaves != n.leaves || !(n.last_ahead == pos))) {
          history_.push_back({ts, id, pos, n.leaves, QueueStatus::Resting});
          n.last_ahead = pos;
          n.last_leaves = n.leaves;
        }
        ++pos.orders;
        pos.qty += n.leaves;
      }
    }
    dirty_.clear();
  }

  std::unordered_map<OrderId, Node> orders_;
  Levels bids_;
  Levels asks_;
  std::vector<std::pair<Side, std::int64_t>> dirty_;
  std::unordered_set<OrderId> wanted_;
  std::vector<QueueSample> history_;
  Timestamp now_{0};
};

// ---------------------------------------------------------------- fan-out

// Forwards every event to each listener, in the order they were added.
class Fanout final : public Listener {
 public:
  Fanout() = default;
  explicit Fanout(std::vector<Listener*> to) : to_(std::move(to)) {}
  void add(Listener* l) { to_.push_back(l); }
  void on_trade(const Trade& t) override {
    for (Listener* l : to_) l->on_trade(t);
  }
  void on_level_update(const LevelUpdate& u) override {
    for (Listener* l : to_) l->on_level_update(u);
  }
  void on_report(const ExecReport& r) override {
    for (Listener* l : to_) l->on_report(r);
  }

 private:
  std::vector<Listener*> to_;
};

}  // namespace lob::analytics
