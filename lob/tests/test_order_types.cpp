// Matching behaviour of each order type: limit, market, IOC, FOK, post-only.
// Seeded book (see BookTest::seed_book):
//   asks  101: [1:10, 2:20]   102: [3:30]   103: [4:40]
//   bids   99: [5:10, 6:20]    98: [7:30]

#include <vector>

#include "book_fixture.hpp"

namespace lob::testing_helpers {
namespace {

using Limit = BookTest;
using Market = BookTest;
using Ioc = BookTest;
using Fok = BookTest;
using PostOnly = BookTest;

// ---------------------------------------------------------------- limit

TEST_F(Limit, RestsWhenItDoesNotCross) {
  seed_book();
  EXPECT_EQ(limit(10, B, 100, 5), filled(0, 5));
  EXPECT_EQ(book.best_bid(), Price{100});
  EXPECT_TRUE(log.trades.empty());
  ASSERT_EQ(log.reports.size(), 1u);
  EXPECT_EQ(log.reports[0], (ExecReport{10, ExecType::New, Reason::None, B, Price{100}, 0, 5, 0}));
  EXPECT_EQ(log.updates, (std::vector<LevelUpdate>{{B, Price{100}, 5, 1, 0}}));
}

TEST_F(Limit, TradesAtTheMakerPrice) {
  seed_book();
  // A buy limit at 105 crossing an ask at 101 gets the 101 price (price improvement).
  EXPECT_EQ(limit(10, B, 105, 10), filled(10));
  ASSERT_EQ(log.trades.size(), 1u);
  EXPECT_EQ(log.trades[0], (Trade{10, 1, B, Price{101}, 10, 0}));
}

TEST_F(Limit, CrossesAtAnEqualPrice) {
  seed_book();
  EXPECT_EQ(limit(10, S, 99, 5), filled(5));
  EXPECT_EQ(log.trades[0].maker, 5u);
}

TEST_F(Limit, FifoWithinALevel) {
  seed_book();
  EXPECT_EQ(limit(10, B, 101, 15), filled(15));
  ASSERT_EQ(log.trades.size(), 2u);
  EXPECT_EQ(log.trades[0], (Trade{10, 1, B, Price{101}, 10, 0}));  // oldest first, in full
  EXPECT_EQ(log.trades[1], (Trade{10, 2, B, Price{101}, 5, 0}));   // then the next, partly
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{2}));
  EXPECT_EQ(book.find(2)->qty, 15);  // a partial fill keeps its place at the head
}

TEST_F(Limit, SweepsLevelsInPriceOrderThenRestsTheRemainder) {
  seed_book();
  // Buy 70 up to 102: takes 101 (30) and 102 (30), never touches 103, rests 10 at 102.
  EXPECT_EQ(limit(10, B, 102, 70), filled(60, 10));
  std::vector<OrderId> makers;
  for (const Trade& t : log.trades) makers.push_back(t.maker);
  EXPECT_EQ(makers, (std::vector<OrderId>{1, 2, 3}));
  EXPECT_EQ(book.best_bid(), Price{102});
  EXPECT_EQ(book.best_ask(), Price{103});
  EXPECT_EQ(book.find(10)->qty, 10);
}

TEST_F(Limit, NeverTradesThroughItsLimit) {
  seed_book();
  EXPECT_EQ(limit(10, S, 99, 100), filled(30, 70));  // takes 99 only; 98 is below the limit
  EXPECT_EQ(book.best_bid(), Price{98});
  EXPECT_EQ(book.best_ask(), Price{99});
}

// ---------------------------------------------------------------- market

TEST_F(Market, SweepsAsManyLevelsAsNeeded) {
  seed_book();
  EXPECT_EQ(market(10, B, 65), filled(65));
  EXPECT_EQ(log.trades.back(), (Trade{10, 4, B, Price{103}, 5, 0}));
  EXPECT_EQ(book.best_ask(), Price{103});
  EXPECT_EQ(book.best(S)->qty, 35);
}

TEST_F(Market, RemainderIsCancelledNeverRests) {
  seed_book();
  EXPECT_EQ(market(10, S, 100), filled(60, 0, 40));
  EXPECT_FALSE(book.best_bid());
  EXPECT_FALSE(book.find(10));
  EXPECT_EQ(book.best_ask(), Price{101});  // nothing landed on the ask side
  EXPECT_EQ(log.reports.back(),
            (ExecReport{10, ExecType::Cancelled, Reason::Unfilled, S, Price{}, 40, 0, 0}));
}

TEST_F(Market, AgainstAnEmptySideIsCancelledInFull) {
  EXPECT_EQ(market(1, B, 10), filled(0, 0, 10));
  EXPECT_EQ(book.order_count(), 0u);
  ASSERT_EQ(log.reports.size(), 2u);
  EXPECT_EQ(log.reports[0].type, ExecType::New);
  EXPECT_EQ(log.reports[1].type, ExecType::Cancelled);
}

TEST_F(Market, IgnoresItsPriceField) {
  seed_book();
  // A price that would not cross as a limit is irrelevant to a market order.
  EXPECT_EQ(book.submit(make(10, B, OrderType::Market, 1, 10)), filled(10));
}

// ---------------------------------------------------------------- IOC

TEST_F(Ioc, FillsWhatCrossesAndCancelsTheRest) {
  seed_book();
  EXPECT_EQ(ioc(10, B, 102, 100), filled(60, 0, 40));
  EXPECT_EQ(book.best_ask(), Price{103});  // stopped at its limit
  EXPECT_EQ(book.best_bid(), Price{99});   // nothing rested
  EXPECT_FALSE(book.find(10));
}

TEST_F(Ioc, FullyFilledLeavesNothingToCancel) {
  seed_book();
  EXPECT_EQ(ioc(10, S, 99, 25), filled(25));
  EXPECT_EQ(log.reports.back().type, ExecType::Fill);
  EXPECT_EQ(log.reports.back().leaves, 0);
}

TEST_F(Ioc, ThatDoesNotCrossIsCancelledAndLeavesTheBookAlone) {
  seed_book();
  const auto before_bids = book.depth(B, 10);
  const auto before_asks = book.depth(S, 10);
  EXPECT_EQ(ioc(10, B, 100, 10), filled(0, 0, 10));
  EXPECT_EQ(book.depth(B, 10), before_bids);
  EXPECT_EQ(book.depth(S, 10), before_asks);
  EXPECT_TRUE(log.trades.empty());
  EXPECT_TRUE(log.updates.empty());
}

// ---------------------------------------------------------------- FOK

TEST_F(Fok, FillsCompletelyAcrossLevels) {
  seed_book();
  EXPECT_EQ(fok(10, B, 103, 100), filled(100));
  EXPECT_FALSE(book.best_ask());
  EXPECT_EQ(log.trades.size(), 4u);
}

TEST_F(Fok, ExactlyEnoughLiquidityFills) {
  seed_book();
  EXPECT_EQ(fok(10, B, 102, 60), filled(60));
  EXPECT_EQ(book.best_ask(), Price{103});
}

TEST_F(Fok, KilledWhenNotEnoughWithinItsLimit) {
  seed_book();
  const auto before = book.depth(S, 10);
  // 100 shares exist on the ask, but only 60 are at or below 102.
  EXPECT_EQ(fok(10, B, 102, 61), rejected(Reason::InsufficientLiquidity));
  EXPECT_EQ(book.depth(S, 10), before);
  EXPECT_TRUE(log.trades.empty());
  EXPECT_TRUE(log.updates.empty());
  ASSERT_EQ(log.reports.size(), 1u);
  EXPECT_EQ(log.reports[0].type, ExecType::Rejected);
  EXPECT_FALSE(book.find(10));
}

TEST_F(Fok, AgainstAnEmptySideIsKilled) {
  EXPECT_EQ(fok(1, S, 100, 1), rejected(Reason::InsufficientLiquidity));
}

// ---------------------------------------------------------------- post-only

TEST_F(PostOnly, RestsWhenPassive) {
  seed_book();
  EXPECT_EQ(post_only(10, S, 100, 5), filled(0, 5));
  EXPECT_EQ(book.best_ask(), Price{100});
  EXPECT_EQ(book.find(10)->type, OrderType::PostOnly);
  EXPECT_TRUE(log.trades.empty());
}

TEST_F(PostOnly, RejectedIfItWouldCross) {
  seed_book();
  const auto before = book.depth(S, 10);
  EXPECT_EQ(post_only(10, B, 101, 5), rejected(Reason::WouldCross));  // touching the ask
  EXPECT_EQ(post_only(11, B, 110, 5), rejected(Reason::WouldCross));  // through it
  EXPECT_EQ(book.depth(S, 10), before);
  EXPECT_FALSE(book.find(10));
  EXPECT_TRUE(log.trades.empty());
}

TEST_F(PostOnly, RestsAgainstAnEmptyOppositeSide) {
  EXPECT_EQ(post_only(1, B, 1'000'000, 5), filled(0, 5));
}

// ---------------------------------------------------------------- accounting, all types

// Every accepted order is fully accounted for: filled + resting + cancelled == quantity sent,
// and the trades printed add up to the filled quantity.
TEST_F(BookTest, QuantityIsConservedForEveryOrderType) {
  struct Case {
    OrderType type;
    Side side;
    std::int64_t px;
    Quantity qty;
  };
  const std::vector<Case> cases = {
      {OrderType::Limit, B, 102, 45},   {OrderType::Limit, S, 98, 100},
      {OrderType::Market, B, 0, 500},   {OrderType::IOC, S, 99, 50},
      {OrderType::FOK, B, 103, 100},    {OrderType::PostOnly, S, 104, 10},
  };
  OrderId id = 100;
  for (const Case& c : cases) {
    seed_book();
    const Quantity bids = book.resting_qty(B), asks = book.resting_qty(S);
    const Result r = book.submit(make(++id, c.side, c.type, c.px, c.qty));
    ASSERT_TRUE(r.accepted) << to_string(c.type);
    EXPECT_EQ(r.filled + r.resting + r.cancelled, c.qty) << to_string(c.type);
    Quantity traded = 0;
    for (const Trade& t : log.trades) traded += t.qty;
    EXPECT_EQ(traded, r.filled) << to_string(c.type);
    // Liquidity taken from the far side equals the fill; the near side gains what rested.
    const Quantity far_before = c.side == B ? asks : bids;
    const Quantity near_before = c.side == B ? bids : asks;
    EXPECT_EQ(book.resting_qty(opposite(c.side)), far_before - r.filled);
    EXPECT_EQ(book.resting_qty(c.side), near_before + r.resting);
    // Clear the book for the next case.
    for (OrderId o = 1; o <= id; ++o) book.cancel(o);
    log.clear();
  }
}

}  // namespace
}  // namespace lob::testing_helpers
