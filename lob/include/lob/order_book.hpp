#pragma once

// Single-threaded price-time priority limit order book and matching engine.
//
// Layout
//   - Each side holds its price levels best price first, in one of two stores (levels.hpp):
//     an ordered map (OrderBook) or a tick-indexed array (ArrayOrderBook). The engine code
//     is shared; step 4 benchmarks the two.
//   - Each Level is an intrusive doubly linked FIFO of resting orders: append at the tail,
//     match from the head, unlink anywhere in O(1).
//   - Order nodes come from a fixed-address pool, and an open-addressing id map points
//     straight at them, so cancel and modify find their order in O(1) without searching.
//
// Priority rules
//   - Better price first; at one price, earlier arrival first.
//   - A modify that only reduces size keeps the order's place in the queue. A price change
//     or a size increase is a cancel-replace: the order goes to the back of its new level
//     and, for a limit order re-priced through the spread, trades first.
//   - Trades print at the resting (maker) order's price.
//
// Order types (see Order::type)
//   Limit     match what crosses, rest the remainder
//   Market    match at any price, cancel the remainder (never rests)
//   IOC       match what crosses its limit, cancel the remainder
//   FOK       fill completely within its limit, or reject with nothing traded
//   PostOnly  rest without trading, or reject if it would cross

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "lob/events.hpp"
#include "lob/id_map.hpp"
#include "lob/levels.hpp"
#include "lob/object_pool.hpp"
#include "lob/types.hpp"

namespace lob {

// Outcome of one call. For an accepted order, filled + resting + cancelled == quantity sent.
struct Result {
  bool accepted{false};
  Reason reason{Reason::None};
  Quantity filled{0};     // traded during this call
  Quantity resting{0};    // left on the book after this call
  Quantity cancelled{0};  // removed without trading (IOC/market remainder, cancel)

  friend bool operator==(const Result&, const Result&) = default;
};

struct LevelView {
  Price price{};
  Quantity qty{};
  std::uint32_t count{0};

  friend bool operator==(const LevelView&, const LevelView&) = default;
};

struct OrderView {
  OrderId id{};
  Side side{Side::Buy};
  OrderType type{OrderType::Limit};
  Price price{};
  Quantity qty{};
  Timestamp ts{};

  friend bool operator==(const OrderView&, const OrderView&) = default;
};

template <typename Levels>
class BasicOrderBook {
 public:
  // `listener` may be null (no events). `expected_orders` sizes the pool and id map so a book
  // holding up to that many resting orders never allocates for them. Prices must be positive
  // multiples of `tick_size` (an off-grid price is rejected as InvalidPrice); the array store
  // uses it as its slot width.
  explicit BasicOrderBook(Listener* listener = nullptr, std::size_t expected_orders = 1 << 16,
                          std::int64_t tick_size = 1);

  BasicOrderBook(const BasicOrderBook&) = delete;
  BasicOrderBook& operator=(const BasicOrderBook&) = delete;
  BasicOrderBook(BasicOrderBook&&) = delete;
  BasicOrderBook& operator=(BasicOrderBook&&) = delete;
  ~BasicOrderBook() = default;

  // New order of any type. The price of a Market order is ignored.
  Result submit(const Order& order);

  // Removes a resting order.
  Result cancel(OrderId id, Timestamp ts = 0);

  // Changes a resting order's price and/or open quantity (`new_qty` is the new *open*
  // quantity, not including anything already filled). Reducing size at the same price keeps
  // priority; anything else loses it. A post-only order re-priced through the spread is
  // rejected and left untouched.
  Result modify(OrderId id, Price new_price, Quantity new_qty, Timestamp ts = 0);

  void set_listener(Listener* listener) noexcept { listener_ = listener; }

  // ---- read-only views ----
  [[nodiscard]] std::optional<Price> best_bid() const noexcept;
  [[nodiscard]] std::optional<Price> best_ask() const noexcept;
  [[nodiscard]] std::optional<LevelView> best(Side side) const noexcept;
  // Up to `max_levels` levels of one side, best first.
  [[nodiscard]] std::vector<LevelView> depth(Side side, std::size_t max_levels) const;
  // Ids resting at one price, in priority order (empty if no such level).
  [[nodiscard]] std::vector<OrderId> queue(Side side, Price price) const;
  // L3 snapshot: every resting order on one side in priority order (best price first,
  // oldest first within a price).
  [[nodiscard]] std::vector<OrderView> orders(Side side) const;
  [[nodiscard]] std::optional<OrderView> find(OrderId id) const noexcept;
  [[nodiscard]] std::size_t order_count() const noexcept { return index_.size(); }
  [[nodiscard]] std::size_t level_count(Side side) const noexcept;
  [[nodiscard]] Quantity resting_qty(Side side) const noexcept;
  [[nodiscard]] std::int64_t tick_size() const noexcept { return tick_; }

  // Full structural self-check: uncrossed book, level totals and counts, FIFO sequence
  // order, link integrity, index and pool agreement. Returns "" if sound, else the first
  // violation found. O(orders); meant for tests and debug builds.
  [[nodiscard]] std::string validate() const;

 private:
  using Bids = typename Levels::template Side<detail::HigherFirst>;
  using Asks = typename Levels::template Side<detail::LowerFirst>;

  // Calls f(bids_) or f(asks_). Both instantiations must return the same type.
  template <typename F>
  decltype(auto) on_side(Side side, F&& f) {
    return side == Side::Buy ? std::invoke(f, bids_) : std::invoke(f, asks_);
  }
  template <typename F>
  decltype(auto) on_side(Side side, F&& f) const {
    return side == Side::Buy ? std::invoke(f, bids_) : std::invoke(f, asks_);
  }

  [[nodiscard]] bool would_cross(Side side, Price limit) const noexcept;
  [[nodiscard]] Quantity available(Side side, Price limit, Quantity needed) const noexcept;

  // Trades an incoming order against the opposite side. Returns the quantity filled.
  Quantity match(OrderId taker, Side side, OrderType type, Price limit, Quantity qty,
                 Timestamp ts);
  template <typename Opp>
  Quantity match_against(Opp& opp, OrderId taker, Side side, OrderType type, Price limit,
                         Quantity qty, Timestamp ts);

  void rest(OrderId id, Side side, OrderType type, Price price, Quantity qty, Timestamp ts);
  void remove(OrderNode* node, Timestamp ts);  // unlink, free, emit the level change
  void reserve_one();  // so the next rest() cannot allocate a node or grow the index

  void emit(const Trade& t) const {
    if (listener_ != nullptr) listener_->on_trade(t);
  }
  void emit(const LevelUpdate& u) const {
    if (listener_ != nullptr) listener_->on_level_update(u);
  }
  void emit(const ExecReport& r) const {
    if (listener_ != nullptr) listener_->on_report(r);
  }
  Result reject(OrderId id, Side side, Price price, Reason reason, Timestamp ts) const;

  [[nodiscard]] bool valid_price(Side side, Price p) const noexcept {
    return on_side(side, [&](const auto& s) { return s.accepts(p); });
  }

  Listener* listener_;
  std::int64_t tick_;
  Bids bids_;
  Asks asks_;
  ObjectPool<OrderNode> pool_;
  IdMap<OrderNode> index_;
  std::uint64_t next_seq_{0};
};

// The two books. Member definitions live in order_book.cpp, instantiated for both.
using OrderBook = BasicOrderBook<MapLevels>;
using ArrayOrderBook = BasicOrderBook<ArrayLevels>;
extern template class BasicOrderBook<MapLevels>;
extern template class BasicOrderBook<ArrayLevels>;

}  // namespace lob
