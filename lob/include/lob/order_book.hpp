#pragma once

// Single-threaded price-time priority limit order book and matching engine.
//
// Layout
//   - Each side is an ordered map from price to Level, best price first. (Phase 2 step 4
//     benchmarks this against a tick-indexed array.)
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
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "lob/events.hpp"
#include "lob/id_map.hpp"
#include "lob/object_pool.hpp"
#include "lob/types.hpp"

namespace lob {

struct Level;

struct OrderNode {
  OrderId id{};
  Price price{};
  Quantity qty{};      // open (unfilled) quantity
  Timestamp ts{};      // event time of the message that gave the order its current priority
  std::uint64_t seq{}; // arrival sequence number; strictly increasing head -> tail in a level
  Side side{Side::Buy};
  OrderType type{OrderType::Limit};
  OrderNode* prev{nullptr};
  OrderNode* next{nullptr};
  Level* level{nullptr};
};

struct Level {
  Price price{};
  Quantity total{0};        // sum of open quantity of the orders below
  std::uint32_t count{0};
  OrderNode* head{nullptr};  // oldest: matched first
  OrderNode* tail{nullptr};  // newest
};

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

namespace detail {

struct HigherFirst {
  constexpr bool operator()(Price a, Price b) const noexcept { return a > b; }
};
struct LowerFirst {
  constexpr bool operator()(Price a, Price b) const noexcept { return a < b; }
};

// One side of the book: price levels ordered best-first.
template <typename BestFirst>
class BookSide {
 public:
  using Map = std::map<Price, Level, BestFirst>;

  [[nodiscard]] bool empty() const noexcept { return levels_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return levels_.size(); }
  [[nodiscard]] Level& best() noexcept { return levels_.begin()->second; }
  [[nodiscard]] const Level& best() const noexcept { return levels_.begin()->second; }

  // Existing level at `price`, or a new empty one. Map nodes never move, so the returned
  // reference (and Level* kept in order nodes) stays valid until the level is erased.
  Level& get_or_create(Price price) {
    auto [it, inserted] = levels_.try_emplace(price);
    if (inserted) it->second.price = price;
    return it->second;
  }

  [[nodiscard]] const Level* find(Price price) const {
    auto it = levels_.find(price);
    return it == levels_.end() ? nullptr : &it->second;
  }

  void erase(Price price) { levels_.erase(price); }

  [[nodiscard]] typename Map::const_iterator begin() const noexcept { return levels_.begin(); }
  [[nodiscard]] typename Map::const_iterator end() const noexcept { return levels_.end(); }

 private:
  Map levels_;
};

}  // namespace detail

class OrderBook {
 public:
  // `listener` may be null (no events). `expected_orders` sizes the pool and id map so a book
  // holding up to that many resting orders never allocates for them.
  explicit OrderBook(Listener* listener = nullptr, std::size_t expected_orders = 1 << 16);

  OrderBook(const OrderBook&) = delete;
  OrderBook& operator=(const OrderBook&) = delete;
  OrderBook(OrderBook&&) = delete;
  OrderBook& operator=(OrderBook&&) = delete;
  ~OrderBook() = default;

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

  // Full structural self-check: uncrossed book, level totals and counts, FIFO sequence
  // order, link integrity, index and pool agreement. Returns "" if sound, else the first
  // violation found. O(orders); meant for tests and debug builds.
  [[nodiscard]] std::string validate() const;

 private:
  using Bids = detail::BookSide<detail::HigherFirst>;
  using Asks = detail::BookSide<detail::LowerFirst>;

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

  Listener* listener_;
  Bids bids_;
  Asks asks_;
  ObjectPool<OrderNode> pool_;
  IdMap<OrderNode> index_;
  std::uint64_t next_seq_{0};
};

}  // namespace lob
