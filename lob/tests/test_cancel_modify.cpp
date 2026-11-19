// Cancel and modify, including the queue-priority rules:
// size down at the same price keeps priority; a price change or a size increase loses it.

#include <vector>

#include "book_fixture.hpp"

namespace lob::testing_helpers {
namespace {

using Cancel = BookTest;
using Modify = BookTest;

// ---------------------------------------------------------------- cancel

TEST_F(Cancel, RemovesTheOrderAndAnEmptiedLevel) {
  seed_book();
  EXPECT_EQ(book.cancel(3, 77), filled(0, 0, 30));
  EXPECT_FALSE(book.find(3));
  EXPECT_EQ(book.level_count(S), 2u);
  EXPECT_EQ(log.updates, (std::vector<LevelUpdate>{{S, Price{102}, 0, 0, 77}}));
  EXPECT_EQ(log.reports.back(),
            (ExecReport{3, ExecType::Cancelled, Reason::UserCancel, S, Price{102}, 30, 0, 77}));
}

TEST_F(Cancel, FromTheMiddleOfAQueueKeepsTheOthersInOrder) {
  limit(1, B, 99, 10);
  limit(2, B, 99, 10);
  limit(3, B, 99, 10);
  EXPECT_TRUE(book.cancel(2).accepted);
  EXPECT_EQ(book.queue(B, Price{99}), (std::vector<OrderId>{1, 3}));
  EXPECT_EQ(book.best(B), (LevelView{Price{99}, 20, 2}));
  EXPECT_TRUE(book.cancel(3).accepted);  // tail
  EXPECT_TRUE(book.cancel(1).accepted);  // head, and last
  EXPECT_FALSE(book.best_bid());
}

TEST_F(Cancel, UnknownIdIsRejected) {
  seed_book();
  EXPECT_EQ(book.cancel(999), rejected(Reason::UnknownOrder));
  EXPECT_EQ(book.order_count(), 7u);
}

TEST_F(Cancel, TwiceIsRejectedTheSecondTime) {
  seed_book();
  EXPECT_TRUE(book.cancel(5).accepted);
  EXPECT_EQ(book.cancel(5), rejected(Reason::UnknownOrder));
}

TEST_F(Cancel, AfterAFullFillIsRejected) {
  seed_book();
  market(10, B, 10);  // fills id 1 completely
  EXPECT_EQ(book.cancel(1), rejected(Reason::UnknownOrder));
}

TEST_F(Cancel, AfterAPartialFillRemovesWhatIsLeft) {
  seed_book();
  market(10, B, 4);  // id 1: 10 -> 6
  EXPECT_EQ(book.cancel(1), filled(0, 0, 6));
}

// ---------------------------------------------------------------- modify: priority

TEST_F(Modify, SizeDownKeepsPriority) {
  limit(1, S, 101, 10);
  limit(2, S, 101, 10);
  EXPECT_EQ(book.modify(1, Price{101}, 4), filled(0, 4));
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{1, 2}));
  EXPECT_EQ(book.best(S), (LevelView{Price{101}, 14, 2}));
  market(3, B, 5);
  EXPECT_EQ(log.trades.front().maker, 1u);  // still first in line
}

TEST_F(Modify, SizeUpLosesPriority) {
  limit(1, S, 101, 10);
  limit(2, S, 101, 10);
  EXPECT_EQ(book.modify(1, Price{101}, 15), filled(0, 15));
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{2, 1}));
  EXPECT_EQ(book.best(S), (LevelView{Price{101}, 25, 2}));
}

TEST_F(Modify, PriceChangeLosesPriorityEvenWhenMovedBack) {
  limit(1, B, 99, 10);
  limit(2, B, 99, 10);
  book.modify(1, Price{98}, 10);
  book.modify(1, Price{99}, 10);
  EXPECT_EQ(book.queue(B, Price{99}), (std::vector<OrderId>{2, 1}));
}

TEST_F(Modify, KeepsTheOrderIdAndType) {
  post_only(1, B, 95, 10);
  book.modify(1, Price{96}, 12, 500);
  const auto o = book.find(1);
  ASSERT_TRUE(o);
  EXPECT_EQ(*o, (OrderView{1, B, OrderType::PostOnly, Price{96}, 12, 500}));
}

// ---------------------------------------------------------------- modify: re-price through

TEST_F(Modify, LimitRepricedThroughTheSpreadTradesThenRests) {
  seed_book();
  // Bid 5 (10 @ 99) re-priced to 101 for 40: takes the 30 at 101, rests 10 at 101.
  EXPECT_EQ(book.modify(5, Price{101}, 40), filled(30, 10));
  EXPECT_EQ(log.trades.size(), 2u);
  EXPECT_EQ(log.trades[0].taker, 5u);
  EXPECT_EQ(book.best_bid(), Price{101});
  EXPECT_EQ(book.best_ask(), Price{102});
  EXPECT_EQ(book.find(5)->qty, 10);
  EXPECT_EQ(book.queue(B, Price{99}), (std::vector<OrderId>{6}));  // left its old level
}

TEST_F(Modify, LimitRepricedThroughCanFillCompletely) {
  seed_book();
  EXPECT_EQ(book.modify(7, Price{101}, 30), filled(30));
  EXPECT_FALSE(book.find(7));
}

TEST_F(Modify, PostOnlyRepricedThroughIsRejectedAndUntouched) {
  seed_book();
  post_only(10, B, 100, 5);
  limit(11, B, 100, 5);
  log.clear();
  EXPECT_EQ(book.modify(10, Price{101}, 5), rejected(Reason::WouldCross));
  EXPECT_EQ(book.queue(B, Price{100}), (std::vector<OrderId>{10, 11}));  // priority intact
  EXPECT_EQ(book.find(10)->price, Price{100});
  EXPECT_TRUE(log.trades.empty());
  EXPECT_TRUE(log.updates.empty());
}

TEST_F(Modify, PostOnlySizeUpAtTheSamePriceIsAllowed) {
  seed_book();
  post_only(10, S, 100, 5);
  EXPECT_EQ(book.modify(10, Price{100}, 8), filled(0, 8));
}

// ---------------------------------------------------------------- modify: edge cases

TEST_F(Modify, SameValuesIsANoOp) {
  seed_book();
  EXPECT_EQ(book.modify(2, Price{101}, 20), filled(0, 20));
  EXPECT_TRUE(log.reports.empty());
  EXPECT_TRUE(log.updates.empty());
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{1, 2}));
}

TEST_F(Modify, RejectsBadArgumentsAndUnknownIds) {
  seed_book();
  EXPECT_EQ(book.modify(1, Price{101}, 0), rejected(Reason::InvalidQuantity));
  EXPECT_EQ(book.modify(1, Price{101}, -1), rejected(Reason::InvalidQuantity));
  EXPECT_EQ(book.modify(1, Price{0}, 5), rejected(Reason::InvalidPrice));
  EXPECT_EQ(book.modify(999, Price{101}, 5), rejected(Reason::UnknownOrder));
  EXPECT_EQ(book.find(1)->qty, 10);
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{1, 2}));
}

TEST_F(Modify, AfterAPartialFillWorksOnTheOpenQuantity) {
  seed_book();
  market(10, B, 4);  // id 1: 10 open -> 6 open
  EXPECT_EQ(book.modify(1, Price{101}, 5), filled(0, 5));  // 6 -> 5 open: a size-down
  EXPECT_EQ(book.queue(S, Price{101}), (std::vector<OrderId>{1, 2}));
}

}  // namespace
}  // namespace lob::testing_helpers
