#pragma once

// Shared test fixture: a book wired to an EventLog, order helpers, and a structural
// self-check after every test.

#include <gtest/gtest.h>

#include <ostream>

#include "lob/events.hpp"
#include "lob/order_book.hpp"

namespace lob {

// Readable gtest failure messages.
inline std::ostream& operator<<(std::ostream& os, Price p) { return os << p.ticks; }
inline std::ostream& operator<<(std::ostream& os, Side s) { return os << to_string(s); }
inline std::ostream& operator<<(std::ostream& os, Reason r) { return os << to_string(r); }
inline std::ostream& operator<<(std::ostream& os, ExecType t) { return os << to_string(t); }
inline std::ostream& operator<<(std::ostream& os, const Result& r) {
  return os << "{accepted=" << r.accepted << " reason=" << r.reason << " filled=" << r.filled
            << " resting=" << r.resting << " cancelled=" << r.cancelled << "}";
}
inline std::ostream& operator<<(std::ostream& os, const LevelView& l) {
  return os << "{" << l.price << " x " << l.qty << " (" << l.count << ")}";
}
inline std::ostream& operator<<(std::ostream& os, const Trade& t) {
  return os << "{taker=" << t.taker << " maker=" << t.maker << " " << t.taker_side << " "
            << t.qty << "@" << t.price << "}";
}

namespace testing_helpers {

inline constexpr Side B = Side::Buy;
inline constexpr Side S = Side::Sell;

inline Order make(OrderId id, Side side, OrderType type, std::int64_t px, Quantity qty,
                  Timestamp ts = 0) {
  return Order{id, side, type, Price{px}, qty, ts};
}

class BookTest : public ::testing::Test {
 protected:
  void TearDown() override { EXPECT_EQ(book.validate(), ""); }

  Result limit(OrderId id, Side side, std::int64_t px, Quantity qty) {
    return book.submit(make(id, side, OrderType::Limit, px, qty));
  }
  Result market(OrderId id, Side side, Quantity qty) {
    return book.submit(make(id, side, OrderType::Market, 0, qty));
  }
  Result ioc(OrderId id, Side side, std::int64_t px, Quantity qty) {
    return book.submit(make(id, side, OrderType::IOC, px, qty));
  }
  Result fok(OrderId id, Side side, std::int64_t px, Quantity qty) {
    return book.submit(make(id, side, OrderType::FOK, px, qty));
  }
  Result post_only(OrderId id, Side side, std::int64_t px, Quantity qty) {
    return book.submit(make(id, side, OrderType::PostOnly, px, qty));
  }

  // Asks at 101 (ids 1,2), 102 (id 3), 103 (id 4); bids at 99 (ids 5,6), 98 (id 7).
  void seed_book() {
    limit(1, S, 101, 10);
    limit(2, S, 101, 20);
    limit(3, S, 102, 30);
    limit(4, S, 103, 40);
    limit(5, B, 99, 10);
    limit(6, B, 99, 20);
    limit(7, B, 98, 30);
    log.clear();
  }

  static Result filled(Quantity f, Quantity resting = 0, Quantity cancelled = 0) {
    return Result{.accepted = true, .filled = f, .resting = resting, .cancelled = cancelled};
  }
  static Result rejected(Reason why) { return Result{.accepted = false, .reason = why}; }

  EventLog log;
  OrderBook book{&log, 64};  // small so the pool and index are forced to grow in busy tests
};

}  // namespace testing_helpers
}  // namespace lob
