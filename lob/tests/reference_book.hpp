#pragma once

// Reference model of the matching rules, written to be obviously right rather than fast:
// every resting order sits in one flat vector, and the next order to trade against is found
// by a linear scan for the best price, then the earliest arrival. The differential tests
// run the real engine and this model side by side and require identical results, trades
// and full L3 state after every call.
//
// `Mutation` plants one known bug (test-only). The property suite must catch every one,
// which is the evidence that the properties have teeth.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "lob/events.hpp"
#include "lob/order_book.hpp"
#include "lob/types.hpp"

namespace lob::testing_helpers {

enum class Mutation : std::uint8_t {
  None,
  LifoQueue,                  // newest order at a price trades first
  WorsePriceFirst,            // among crossing prices, the worst trades first
  TradeAtTakerPrice,          // trades print at the incoming order's limit
  TradeThroughLimit,          // matching tolerates one tick through the limit
  StopAfterOneLevel,          // matching never moves past the first price level
  FokFillsPartially,          // fill-or-kill skips its liquidity check
  PostOnlyTakesAtEqualPrice,  // post-only cross check uses > instead of >=
  IocRestsRemainder,          // IOC remainder rests like a limit order
  SizeDownLosesPriority,      // a size reduction sends the order to the back
  SizeUpKeepsPriority,        // a size increase keeps the order's place
  RepriceIgnoresCross,        // a limit re-priced through the spread rests without trading
  RestsFullQuantity,          // after a partial fill the full original size rests
};

// One call's observable outcome.
struct Outcome {
  Result result;
  std::vector<Trade> trades;
};

// Full L3 state: every resting order, best price first, oldest first within a price.
struct BookState {
  std::vector<OrderView> bids;
  std::vector<OrderView> asks;

  [[nodiscard]] const std::vector<OrderView>& side(Side s) const {
    return s == Side::Buy ? bids : asks;
  }
  friend bool operator==(const BookState&, const BookState&) = default;
};

class ReferenceBook {
 public:
  explicit ReferenceBook(Mutation m = Mutation::None) : m_(m) {}

  Outcome submit(const Order& o) {
    const bool priced = o.type != OrderType::Market;
    const Price limit = priced ? o.price : Price{};
    if (o.qty <= 0) return reject(Reason::InvalidQuantity);
    if (priced && o.price.ticks <= 0) return reject(Reason::InvalidPrice);
    if (find(o.id) != npos) return reject(Reason::DuplicateOrderId);
    if (o.type == OrderType::PostOnly && would_cross(o.side, o.price)) {
      return reject(Reason::WouldCross);
    }
    if (o.type == OrderType::FOK && m_ != Mutation::FokFillsPartially &&
        available(o.side, o.price) < o.qty) {
      return reject(Reason::InsufficientLiquidity);
    }
    Outcome out;
    out.result.accepted = true;
    Quantity left = o.qty;
    match(o.id, o.side, o.type, limit, left, o.ts, out.trades);
    out.result.filled = o.qty - left;
    if (left > 0) {
      const bool rests = o.type == OrderType::Limit || o.type == OrderType::PostOnly ||
                         (o.type == OrderType::IOC && m_ == Mutation::IocRestsRemainder);
      if (rests) {
        const Quantity q = m_ == Mutation::RestsFullQuantity ? o.qty : left;
        add(o.id, o.side, o.type, o.price, q, o.ts);
        out.result.resting = q;
      } else {
        out.result.cancelled = left;
      }
    }
    return out;
  }

  Outcome cancel(OrderId id, Timestamp /*ts*/) {
    const std::size_t i = find(id);
    if (i == npos) return reject(Reason::UnknownOrder);
    Outcome out;
    out.result = Result{.accepted = true, .cancelled = orders_[i].v.qty};
    orders_.erase(orders_.begin() + static_cast<std::ptrdiff_t>(i));
    return out;
  }

  Outcome modify(OrderId id, Price price, Quantity qty, Timestamp ts) {
    const std::size_t i = find(id);
    if (i == npos) return reject(Reason::UnknownOrder);
    if (qty <= 0) return reject(Reason::InvalidQuantity);
    if (price.ticks <= 0) return reject(Reason::InvalidPrice);
    Rest& r = orders_[i];
    Outcome out;
    out.result.accepted = true;
    if (price == r.v.price && qty <= r.v.qty) {  // size-down or no-op: keeps priority
      if (qty < r.v.qty && m_ == Mutation::SizeDownLosesPriority) r.seq = ++seq_;
      r.v.qty = qty;
      out.result.resting = qty;
      return out;
    }
    if (price == r.v.price && m_ == Mutation::SizeUpKeepsPriority) {
      r.v.qty = qty;
      out.result.resting = qty;
      return out;
    }
    if (r.v.type == OrderType::PostOnly && price != r.v.price && would_cross(r.v.side, price)) {
      return reject(Reason::WouldCross);
    }
    const OrderView old = r.v;
    orders_.erase(orders_.begin() + static_cast<std::ptrdiff_t>(i));
    Quantity left = qty;
    if (m_ != Mutation::RepriceIgnoresCross) {
      match(id, old.side, old.type, price, left, ts, out.trades);
    }
    out.result.filled = qty - left;
    if (left > 0) {
      add(id, old.side, old.type, price, left, ts);
      out.result.resting = left;
    }
    return out;
  }

  [[nodiscard]] BookState state() const {
    std::vector<Rest> sorted = orders_;
    // The specification's priority order, independent of any mutation.
    std::sort(sorted.begin(), sorted.end(), [](const Rest& a, const Rest& b) {
      if (a.v.side != b.v.side) return a.v.side == Side::Buy;
      if (a.v.price != b.v.price) return better(a.v.side, a.v.price, b.v.price);
      return a.seq < b.seq;
    });
    BookState st;
    for (const Rest& r : sorted) (r.v.side == Side::Buy ? st.bids : st.asks).push_back(r.v);
    return st;
  }

 private:
  struct Rest {
    OrderView v;
    std::uint64_t seq{};
  };
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);

  static Outcome reject(Reason why) {
    Outcome out;
    out.result = Result{.accepted = false, .reason = why};
    return out;
  }

  [[nodiscard]] std::size_t find(OrderId id) const {
    for (std::size_t i = 0; i < orders_.size(); ++i) {
      if (orders_[i].v.id == id) return i;
    }
    return npos;
  }

  [[nodiscard]] bool crosses_limit(Side taker, Price limit, Price resting) const {
    if (m_ == Mutation::TradeThroughLimit) {
      return taker == Side::Buy ? limit.ticks + 1 >= resting.ticks
                                : limit.ticks - 1 <= resting.ticks;
    }
    return crosses(taker, limit, resting);
  }

  [[nodiscard]] bool would_cross(Side taker, Price limit) const {
    for (const Rest& r : orders_) {
      if (r.v.side == taker) continue;
      const bool hit = m_ == Mutation::PostOnlyTakesAtEqualPrice
                           ? (taker == Side::Buy ? limit > r.v.price : limit < r.v.price)
                           : crosses(taker, limit, r.v.price);
      if (hit) return true;
    }
    return false;
  }

  [[nodiscard]] Quantity available(Side taker, Price limit) const {
    Quantity sum = 0;
    for (const Rest& r : orders_) {
      if (r.v.side != taker && crosses_limit(taker, limit, r.v.price)) sum += r.v.qty;
    }
    return sum;
  }

  // Index of the next resting order an incoming order should trade against, or npos.
  [[nodiscard]] std::size_t next_maker(Side taker, OrderType type, Price limit) const {
    std::size_t pick = npos;
    for (std::size_t i = 0; i < orders_.size(); ++i) {
      const Rest& c = orders_[i];
      if (c.v.side == taker) continue;
      if (type != OrderType::Market && !crosses_limit(taker, limit, c.v.price)) continue;
      if (pick == npos) {
        pick = i;
        continue;
      }
      const Rest& p = orders_[pick];
      const Side s = c.v.side;
      if (c.v.price != p.v.price) {
        const bool c_first = m_ == Mutation::WorsePriceFirst ? better(s, p.v.price, c.v.price)
                                                             : better(s, c.v.price, p.v.price);
        if (c_first) pick = i;
      } else if (m_ == Mutation::LifoQueue ? c.seq > p.seq : c.seq < p.seq) {
        pick = i;
      }
    }
    return pick;
  }

  void match(OrderId taker, Side side, OrderType type, Price limit, Quantity& left,
             Timestamp ts, std::vector<Trade>& trades) {
    std::optional<Price> first_level;
    while (left > 0) {
      const std::size_t i = next_maker(side, type, limit);
      if (i == npos) break;
      Rest& mk = orders_[i];
      if (m_ == Mutation::StopAfterOneLevel) {
        if (first_level && mk.v.price != *first_level) break;
        first_level = mk.v.price;
      }
      const Quantity q = std::min(left, mk.v.qty);
      const Price px =
          (m_ == Mutation::TradeAtTakerPrice && type != OrderType::Market) ? limit : mk.v.price;
      trades.push_back(Trade{taker, mk.v.id, side, px, q, ts});
      mk.v.qty -= q;
      left -= q;
      if (mk.v.qty == 0) orders_.erase(orders_.begin() + static_cast<std::ptrdiff_t>(i));
    }
  }

  void add(OrderId id, Side side, OrderType type, Price price, Quantity qty, Timestamp ts) {
    orders_.push_back(Rest{OrderView{id, side, type, price, qty, ts}, ++seq_});
  }

  Mutation m_;
  std::vector<Rest> orders_;
  std::uint64_t seq_{0};
};

}  // namespace lob::testing_helpers
