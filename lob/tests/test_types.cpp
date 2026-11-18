#include <gtest/gtest.h>

#include "lob/types.hpp"

namespace {

using lob::better;
using lob::crosses;
using lob::opposite;
using lob::Price;
using lob::Side;

// Compile-time checks: these helpers sit on the matching hot path and must stay constexpr.
static_assert(opposite(Side::Buy) == Side::Sell);
static_assert(crosses(Side::Buy, Price{101}, Price{100}));
static_assert(!better(Side::Sell, Price{101}, Price{100}));

TEST(Side, OppositeIsAnInvolution) {
  for (Side s : {Side::Buy, Side::Sell}) {
    EXPECT_NE(opposite(s), s);
    EXPECT_EQ(opposite(opposite(s)), s);
  }
}

TEST(Price, OrdersByTicks) {
  EXPECT_LT(Price{100}, Price{101});
  EXPECT_EQ(Price{5}, Price{5});
}

TEST(Price, LobsterScaleRoundTrips) {
  // LOBSTER writes $585.33 as 5853300.
  const Price p = Price::from_lobster(5'853'300);
  EXPECT_DOUBLE_EQ(p.to_dollars(), 585.33);
}

TEST(Crosses, BuyCrossesAtOrAboveAsk) {
  EXPECT_TRUE(crosses(Side::Buy, Price{100}, Price{100}));
  EXPECT_TRUE(crosses(Side::Buy, Price{101}, Price{100}));
  EXPECT_FALSE(crosses(Side::Buy, Price{99}, Price{100}));
}

TEST(Crosses, SellCrossesAtOrBelowBid) {
  EXPECT_TRUE(crosses(Side::Sell, Price{100}, Price{100}));
  EXPECT_TRUE(crosses(Side::Sell, Price{99}, Price{100}));
  EXPECT_FALSE(crosses(Side::Sell, Price{101}, Price{100}));
}

TEST(Better, HigherBidsAndLowerAsksRankFirst) {
  EXPECT_TRUE(better(Side::Buy, Price{101}, Price{100}));
  EXPECT_TRUE(better(Side::Sell, Price{99}, Price{100}));
  EXPECT_FALSE(better(Side::Buy, Price{100}, Price{100}));  // equal price: time decides
}

}  // namespace
