// The event stream: report sequences, trade/fill consistency, and that a subscriber can
// maintain an exact L2 book from LevelUpdate events alone.

#include <map>
#include <random>
#include <vector>

#include "book_fixture.hpp"

namespace lob::testing_helpers {
namespace {

using Events = BookTest;

std::vector<ExecType> types_for(const std::vector<ExecReport>& reports, OrderId id) {
  std::vector<ExecType> out;
  for (const ExecReport& r : reports) {
    if (r.id == id) out.push_back(r.type);
  }
  return out;
}

TEST_F(Events, AggressiveLimitReportsNewFillsThenRests) {
  seed_book();
  book.submit(make(10, B, OrderType::Limit, 101, 35, 9));
  using enum ExecType;
  EXPECT_EQ(types_for(log.reports, 10), (std::vector<ExecType>{New, Fill, Fill}));
  EXPECT_EQ(types_for(log.reports, 1), (std::vector<ExecType>{Fill}));
  // One update for the ask level it emptied, one for the bid level it created.
  EXPECT_EQ(log.updates, (std::vector<LevelUpdate>{{S, Price{101}, 0, 0, 9},
                                                   {B, Price{101}, 5, 1, 9}}));
  for (const Trade& t : log.trades) EXPECT_EQ(t.ts, 9u);
}

TEST_F(Events, FillReportsCarryTheRightLeavesOnBothSides) {
  seed_book();
  book.submit(make(10, B, OrderType::IOC, 101, 15));
  // Trade 1: maker 1 fills 10 (leaves 0), taker leaves 5. Trade 2: maker 2 fills 5 (leaves 15).
  std::vector<ExecReport> fills;
  for (const ExecReport& r : log.reports) {
    if (r.type == ExecType::Fill) fills.push_back(r);
  }
  ASSERT_EQ(fills.size(), 4u);
  EXPECT_EQ(fills[0], (ExecReport{1, ExecType::Fill, Reason::None, S, Price{101}, 10, 0, 0}));
  EXPECT_EQ(fills[1], (ExecReport{10, ExecType::Fill, Reason::None, B, Price{101}, 10, 5, 0}));
  EXPECT_EQ(fills[2], (ExecReport{2, ExecType::Fill, Reason::None, S, Price{101}, 5, 15, 0}));
  EXPECT_EQ(fills[3], (ExecReport{10, ExecType::Fill, Reason::None, B, Price{101}, 5, 0, 0}));
}

TEST_F(Events, ModifyEmitsReplacedAndTheLevelChanges) {
  seed_book();
  book.modify(6, Price{97}, 20, 3);
  EXPECT_EQ(types_for(log.reports, 6), (std::vector<ExecType>{ExecType::Replaced}));
  EXPECT_EQ(log.updates, (std::vector<LevelUpdate>{{B, Price{99}, 10, 1, 3},
                                                   {B, Price{97}, 20, 1, 3}}));
}

TEST_F(Events, NoListenerMeansNoEventsButTheSameBook) {
  OrderBook quiet(nullptr, 8);
  quiet.submit(make(1, S, OrderType::Limit, 101, 10));
  EXPECT_EQ(quiet.submit(make(2, B, OrderType::Limit, 102, 15)),
            (Result{.accepted = true, .filled = 10, .resting = 5}));
  EXPECT_EQ(quiet.best_bid(), Price{102});
  EXPECT_EQ(quiet.validate(), "");
}

// Random order flow of every type plus cancels and modifies, around a drifting mid.
// After every call: (a) an L2 book maintained purely from LevelUpdate events equals the
// engine's own depth, (b) the structural self-check passes. Seeded, so failures reproduce.
TEST_F(Events, LevelUpdateStreamReconstructsTheBookExactly) {
  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int> pick(0, 99);
  std::uniform_int_distribution<std::int64_t> offset(-6, 6);
  std::uniform_int_distribution<Quantity> size(1, 50);

  std::map<Price, Quantity> l2[2];  // [0] = bids, [1] = asks, as seen by a subscriber
  std::size_t seen = 0;
  std::size_t resting_sum = 0;
  std::vector<OrderId> ids;
  OrderId next_id = 1;
  const std::int64_t mid = 10'000;

  for (int step = 0; step < 20'000; ++step) {
    const int p = pick(rng);
    const Side side = (p % 2 == 0) ? B : S;
    // Quotes centred 3 ticks either side of mid: mostly passive, with enough overlap to trade.
    const std::int64_t px = mid + offset(rng) + (side == B ? -3 : 3);
    if (p < 65 || ids.empty()) {
      OrderType t = OrderType::Limit;
      if (p < 8) t = OrderType::Market;
      else if (p < 14) t = OrderType::IOC;
      else if (p < 20) t = OrderType::FOK;
      else if (p < 26) t = OrderType::PostOnly;
      const OrderId id = next_id++;
      if (book.submit(make(id, side, t, px, size(rng), static_cast<Timestamp>(step))).resting > 0) {
        ids.push_back(id);
      }
    } else {
      std::uniform_int_distribution<std::size_t> which(0, ids.size() - 1);
      const OrderId id = ids[which(rng)];
      if (p < 85) {
        book.cancel(id);
      } else {
        const auto o = book.find(id);
        const std::int64_t new_px = o ? o->price.ticks + (p < 93 ? 0 : offset(rng)) : px;
        book.modify(id, Price{new_px > 0 ? new_px : 1}, size(rng));
      }
    }
    // Drop ids that are no longer live (filled or cancelled).
    std::erase_if(ids, [&](OrderId i) { return !book.find(i); });

    for (; seen < log.updates.size(); ++seen) {
      const LevelUpdate& u = log.updates[seen];
      auto& lvls = l2[u.side == B ? 0 : 1];
      if (u.qty == 0) {
        lvls.erase(u.price);
      } else {
        lvls[u.price] = u.qty;
      }
    }
    for (Side s : {B, S}) {
      const auto& lvls = l2[s == B ? 0 : 1];
      const auto d = book.depth(s, lvls.size() + 1);
      ASSERT_EQ(d.size(), lvls.size()) << "step " << step;
      for (const LevelView& v : d) {
        auto it = lvls.find(v.price);
        ASSERT_NE(it, lvls.end()) << "step " << step;
        ASSERT_EQ(it->second, v.qty) << "step " << step;
      }
    }
    ASSERT_EQ(book.validate(), "") << "step " << step;
    resting_sum += book.order_count();
  }
  // The flow must exercise a real book: plenty of trades and a populated, multi-level book.
  EXPECT_GT(log.trades.size(), 2000u);
  EXPECT_GT(resting_sum / 20'000, 20u);
}

}  // namespace
}  // namespace lob::testing_helpers
