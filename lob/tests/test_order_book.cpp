// Book structure, views and input validation.

#include <vector>

#include "book_fixture.hpp"

namespace lob::testing_helpers {
namespace {

using Book = BookTest;

TEST_F(Book, StartsEmpty) {
  EXPECT_EQ(book.order_count(), 0u);
  EXPECT_FALSE(book.best_bid());
  EXPECT_FALSE(book.best_ask());
  EXPECT_TRUE(book.depth(B, 10).empty());
}

TEST_F(Book, DepthIsBestFirstOnBothSides) {
  seed_book();
  EXPECT_EQ(book.depth(S, 10),
            (std::vector<LevelView>{{Price{101}, 30, 2}, {Price{102}, 30, 1}, {Price{103}, 40, 1}}));
  EXPECT_EQ(book.depth(B, 10), (std::vector<LevelView>{{Price{99}, 30, 2}, {Price{98}, 30, 1}}));
  EXPECT_EQ(book.depth(S, 2).size(), 2u);
  EXPECT_EQ(book.best_ask(), Price{101});
  EXPECT_EQ(book.best_bid(), Price{99});
  EXPECT_EQ(book.best(S), (LevelView{Price{101}, 30, 2}));
  EXPECT_EQ(book.order_count(), 7u);
  EXPECT_EQ(book.level_count(S), 3u);
  EXPECT_EQ(book.resting_qty(S), 100);
  EXPECT_EQ(book.resting_qty(B), 60);
}

TEST_F(Book, QueueListsIdsInArrivalOrder) {
  seed_book();
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{1, 2}));
  EXPECT_TRUE(book.queue(S, Price{100}).empty());
}

TEST_F(Book, FindReturnsRestingOrder) {
  book.submit(make(42, B, OrderType::Limit, 99, 15, 1234));
  const auto o = book.find(42);
  ASSERT_TRUE(o);
  EXPECT_EQ(*o, (OrderView{42, B, OrderType::Limit, Price{99}, 15, 1234}));
  EXPECT_FALSE(book.find(43));
}

TEST_F(Book, PoolAndIndexGrowPastInitialSize) {
  // The fixture sizes the book for 64 orders; rest 1,000 on distinct levels.
  for (OrderId id = 1; id <= 1000; ++id) {
    limit(id, B, static_cast<std::int64_t>(id), 1);
  }
  EXPECT_EQ(book.order_count(), 1000u);
  EXPECT_EQ(book.level_count(B), 1000u);
  EXPECT_EQ(book.best_bid(), Price{1000});
}

// ---- validation: every rejection leaves the book exactly as it was

TEST_F(Book, RejectsNonPositiveQuantity) {
  seed_book();
  EXPECT_EQ(limit(10, B, 99, 0), rejected(Reason::InvalidQuantity));
  EXPECT_EQ(market(11, B, -5), rejected(Reason::InvalidQuantity));
  EXPECT_EQ(book.order_count(), 7u);
  ASSERT_EQ(log.reports.size(), 2u);
  EXPECT_EQ(log.reports[0].type, ExecType::Rejected);
  EXPECT_TRUE(log.trades.empty());
  EXPECT_TRUE(log.updates.empty());
}

TEST_F(Book, RejectsNonPositivePriceOnPricedTypes) {
  for (OrderType t : {OrderType::Limit, OrderType::IOC, OrderType::FOK, OrderType::PostOnly}) {
    EXPECT_EQ(book.submit(make(1, B, t, 0, 10)), rejected(Reason::InvalidPrice)) << to_string(t);
    EXPECT_EQ(book.submit(make(1, S, t, -100, 10)), rejected(Reason::InvalidPrice));
  }
  EXPECT_EQ(book.order_count(), 0u);
}

TEST_F(Book, MarketOrderNeedsNoPrice) {
  limit(1, S, 101, 10);
  EXPECT_EQ(market(2, B, 10), filled(10));
}

TEST_F(Book, RejectsDuplicateLiveId) {
  seed_book();
  EXPECT_EQ(limit(3, B, 50, 10), rejected(Reason::DuplicateOrderId));  // 3 rests on the ask
  EXPECT_EQ(market(3, B, 10), rejected(Reason::DuplicateOrderId));     // even if it wouldn't rest
  EXPECT_EQ(book.order_count(), 7u);
  EXPECT_EQ(book.find(3)->side, S);
  EXPECT_TRUE(log.trades.empty());
}

TEST_F(Book, IdCanBeReusedOnceTheOrderIsGone) {
  limit(1, S, 101, 10);
  market(2, B, 10);  // fills id 1 completely
  EXPECT_EQ(limit(1, B, 99, 5), filled(0, 5));
  book.cancel(1);
  EXPECT_EQ(limit(1, S, 105, 5), filled(0, 5));
}

}  // namespace
}  // namespace lob::testing_helpers
