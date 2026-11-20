// The two level stores behind the book (levels.hpp): a tick-indexed array must behave
// exactly like the ordered map, including across page boundaries and when pages are missing.

#include <gtest/gtest.h>

#include <map>
#include <random>
#include <vector>

#include "lob/levels.hpp"
#include "lob/order_book.hpp"

namespace lob {
namespace {

using detail::ArraySide;
using detail::HigherFirst;
using detail::LowerFirst;
using detail::MapSide;

template <typename SideT>
std::vector<std::int64_t> prices(const SideT& s) {
  std::vector<std::int64_t> out;
  s.walk([&](const Level& l) {
    out.push_back(l.price.ticks);
    return true;
  });
  return out;
}

constexpr std::int64_t kPage = ArraySide<LowerFirst>::kPageSize;

TEST(ArrayLevels, WalksBestFirstAcrossPagesOnBothSides) {
  ArraySide<LowerFirst> asks;
  ArraySide<HigherFirst> bids;
  for (std::int64_t p : {5 * kPage + 3, std::int64_t{7}, 2 * kPage, 9 * kPage - 1}) {
    asks.get_or_create(Price{p});
    bids.get_or_create(Price{p});
  }
  EXPECT_EQ(prices(asks), (std::vector<std::int64_t>{7, 2 * kPage, 5 * kPage + 3, 9 * kPage - 1}));
  EXPECT_EQ(prices(bids), (std::vector<std::int64_t>{9 * kPage - 1, 5 * kPage + 3, 2 * kPage, 7}));
  EXPECT_EQ(asks.best().price, Price{7});
  EXPECT_EQ(bids.best().price, Price{9 * kPage - 1});
  EXPECT_EQ(asks.size(), 4u);
}

TEST(ArrayLevels, ErasingTheBestFindsTheNextAcrossMissingPages) {
  ArraySide<LowerFirst> asks;
  asks.get_or_create(Price{10});
  asks.get_or_create(Price{12 * kPage + 5});  // pages 1..11 never exist
  asks.erase(Price{10});
  EXPECT_EQ(asks.best().price, Price{12 * kPage + 5});
  ArraySide<HigherFirst> bids;
  bids.get_or_create(Price{12 * kPage + 5});
  bids.get_or_create(Price{10});
  bids.erase(Price{12 * kPage + 5});
  EXPECT_EQ(bids.best().price, Price{10});
  bids.erase(Price{10});
  EXPECT_TRUE(bids.empty());
}

TEST(ArrayLevels, GrowingDownwardNeverMovesExistingLevels) {
  ArraySide<HigherFirst> bids;
  Level& high = bids.get_or_create(Price{50 * kPage});
  high.total = 42;
  for (std::int64_t pg = 49; pg >= 1; --pg) bids.get_or_create(Price{pg * kPage + 1});
  EXPECT_EQ(&bids.get_or_create(Price{50 * kPage}), &high);  // same address after growth
  EXPECT_EQ(high.total, 42);
  EXPECT_EQ(bids.find(Price{50 * kPage}), &high);
}

TEST(ArrayLevels, AcceptsOnlyPositiveOnGridPricesWithinItsSpan) {
  ArraySide<LowerFirst> asks(100);
  EXPECT_TRUE(asks.accepts(Price{5'853'300}));
  EXPECT_FALSE(asks.accepts(Price{5'853'350}));  // off the one-cent grid
  EXPECT_FALSE(asks.accepts(Price{0}));
  EXPECT_FALSE(asks.accepts(Price{-100}));
  asks.get_or_create(Price{5'853'300});
  const std::int64_t span = ArraySide<LowerFirst>::kMaxPages * kPage * 100;
  EXPECT_TRUE(asks.accepts(Price{5'853'300 + span / 2}));
  EXPECT_FALSE(asks.accepts(Price{5'853'300 + 2 * span}));  // beyond the addressable range
  EXPECT_EQ(asks.find(Price{5'853'350}), nullptr);
}

// Random opens and closes over a wide price range: the array must agree with the map on
// the best level, the full best-first order and every lookup, after every operation.
TEST(ArrayLevels, MatchesTheMapStoreOnRandomOperations) {
  std::mt19937_64 rng(4);
  std::uniform_int_distribution<std::int64_t> px(1, 40 * kPage);  // ~40 pages, mostly sparse
  ArraySide<LowerFirst> a_asks;
  ArraySide<HigherFirst> a_bids;
  MapSide<LowerFirst> m_asks;
  MapSide<HigherFirst> m_bids;
  std::vector<std::int64_t> open;
  for (int step = 0; step < 200'000; ++step) {
    const bool add = open.empty() || rng() % 100 < 55;
    if (add) {
      const Price p{px(rng)};
      if (m_asks.find(p) != nullptr) continue;
      a_asks.get_or_create(p);
      a_bids.get_or_create(p);
      m_asks.get_or_create(p);
      m_bids.get_or_create(p);
      open.push_back(p.ticks);
    } else {
      const std::size_t i = static_cast<std::size_t>(rng() % open.size());
      const Price p{open[i]};
      open[i] = open.back();
      open.pop_back();
      a_asks.erase(p);
      a_bids.erase(p);
      m_asks.erase(p);
      m_bids.erase(p);
    }
    ASSERT_EQ(a_asks.size(), m_asks.size());
    if (!m_asks.empty()) {
      ASSERT_EQ(a_asks.best().price, m_asks.best().price) << "step " << step;
      ASSERT_EQ(a_bids.best().price, m_bids.best().price) << "step " << step;
    }
    if (step % 997 == 0) {
      ASSERT_EQ(prices(a_asks), prices(m_asks)) << "step " << step;
      ASSERT_EQ(prices(a_bids), prices(m_bids)) << "step " << step;
    }
  }
}

// ---- the book on top of it

TEST(ArrayOrderBook, RejectsOffGridAndOutOfRangePrices) {
  ArrayOrderBook book(nullptr, 64, 100);
  EXPECT_TRUE(book.submit(Order{1, Side::Buy, OrderType::Limit, Price{5'853'300}, 10}).accepted);
  EXPECT_EQ(book.submit(Order{2, Side::Buy, OrderType::Limit, Price{5'853'350}, 10}).reason,
            Reason::InvalidPrice);
  EXPECT_EQ(book.submit(Order{3, Side::Buy, OrderType::Limit, Price{1'000'000'000'000}, 10}).reason,
            Reason::InvalidPrice);
  EXPECT_EQ(book.modify(1, Price{5'853'399}, 10).reason, Reason::InvalidPrice);
  EXPECT_TRUE(book.submit(Order{4, Side::Sell, OrderType::Market, Price{}, 10}).accepted);
  EXPECT_EQ(book.validate(), "");
}

TEST(ArrayOrderBook, MapBookEnforcesTheSameTickGrid) {
  OrderBook book(nullptr, 64, 100);
  EXPECT_EQ(book.submit(Order{1, Side::Buy, OrderType::Limit, Price{5'853'350}, 10}).reason,
            Reason::InvalidPrice);
  EXPECT_THROW(OrderBook(nullptr, 64, 0), std::invalid_argument);
}

TEST(ArrayOrderBook, SweepAcrossPagesLeavesTheRightBest) {
  ArrayOrderBook book(nullptr, 64, 1);
  book.submit(Order{1, Side::Sell, OrderType::Limit, Price{100}, 5});
  book.submit(Order{2, Side::Sell, OrderType::Limit, Price{3 * kPage + 7}, 5});
  book.submit(Order{3, Side::Sell, OrderType::Limit, Price{8 * kPage}, 5});
  const Result r = book.submit(Order{4, Side::Buy, OrderType::IOC, Price{3 * kPage + 7}, 10});
  EXPECT_EQ(r.filled, 10);
  EXPECT_EQ(book.best_ask(), Price{8 * kPage});
  EXPECT_EQ(book.validate(), "");
}

}  // namespace
}  // namespace lob
