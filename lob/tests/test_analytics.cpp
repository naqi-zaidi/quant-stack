// Step 6: microstructure analytics, rebuilt from the event stream alone (analytics.hpp).
//
// Hand-computed cases pin each definition. Then a differential test drives both books with the
// step 2 property flows and, after every call, recomputes every statistic by brute force from the
// book's own views: the mirrored L2 and L3 books, the time-weighted sums, the OFI window and
// total (with the formula written out case by case), queue positions and tracked histories. All
// are integers, so all must agree exactly.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lob/analytics.hpp"
#include "lob/digest.hpp"
#include "lob/order_book.hpp"
#include "property_harness.hpp"

namespace lob::testing_helpers {
namespace {

using analytics::DepthProfile;
using analytics::Fanout;
using analytics::OrderFlowImbalance;
using analytics::QueuePosition;
using analytics::QueuePositions;
using analytics::QueueStatus;
using analytics::Quote;
using analytics::SpreadDistribution;

constexpr Side B = Side::Buy;
constexpr Side S = Side::Sell;

Quote quote(std::optional<std::int64_t> bid, Quantity bq, std::optional<std::int64_t> ask,
            Quantity aq) {
  Quote q;
  if (bid) q.bid = Price{*bid};
  q.bid_qty = bq;
  if (ask) q.ask = Price{*ask};
  q.ask_qty = aq;
  return q;
}

// ---------------------------------------------------------------- definitions, by hand

TEST(Analytics, OfiContributionEveryCase) {
  using analytics::ofi_contribution;
  const Quote base = quote(100, 50, 102, 40);
  // Bid side, ask unchanged (ask terms cancel: -40 + 40).
  EXPECT_EQ(ofi_contribution(base, quote(101, 10, 102, 40)), 10);       // bid up: +q_n
  EXPECT_EQ(ofi_contribution(base, quote(100, 70, 102, 40)), 20);       // bid same: q_n - q_n-1
  EXPECT_EQ(ofi_contribution(base, quote(99, 30, 102, 40)), -50);       // bid down: -q_n-1
  // Ask side, bid unchanged.
  EXPECT_EQ(ofi_contribution(base, quote(100, 50, 101, 15)), -15);      // ask down: -q^A_n
  EXPECT_EQ(ofi_contribution(base, quote(100, 50, 102, 25)), 15);       // ask same: -(dq^A)
  EXPECT_EQ(ofi_contribution(base, quote(100, 50, 103, 60)), 40);       // ask up: +q^A_n-1
  // Empty sides: -inf bid / +inf ask with size 0.
  EXPECT_EQ(ofi_contribution(Quote{}, quote(100, 50, std::nullopt, 0)), 50);
  EXPECT_EQ(ofi_contribution(Quote{}, quote(std::nullopt, 0, 102, 40)), -40);
  EXPECT_EQ(ofi_contribution(base, quote(std::nullopt, 0, 102, 40)), -50);  // bid side emptied
  EXPECT_EQ(ofi_contribution(base, quote(100, 50, std::nullopt, 0)), 40);   // ask side emptied
}

TEST(Analytics, OfiWindowTotalAndReset) {
  OrderFlowImbalance ofi(2, /*record=*/true);
  OrderBook book(&ofi);
  book.submit({1, B, OrderType::Limit, Price{100}, 50, 1});  // bid from empty: +50
  book.submit({2, S, OrderType::Limit, Price{102}, 40, 2});  // ask from empty: -40
  book.submit({3, B, OrderType::Limit, Price{99}, 10, 3});   // deeper level: no quote update
  book.submit({4, B, OrderType::Limit, Price{100}, 5, 4});   // bid same, +5
  book.submit({5, B, OrderType::IOC, Price{102}, 40, 5});    // ask queue taken: ask up, +40
  EXPECT_EQ(ofi.updates(), 4u);
  EXPECT_EQ(ofi.total(), 50 - 40 + 5 + 40);
  EXPECT_EQ(ofi.window_sum(), 5 + 40);  // the last two
  EXPECT_EQ(ofi.window_count(), 2u);
  EXPECT_EQ(ofi.last(), 40);
  ASSERT_EQ(ofi.records().size(), 4u);
  EXPECT_EQ(ofi.records()[3].ts, 5u);
  EXPECT_EQ(ofi.quote(), quote(100, 55, std::nullopt, 0));
  EXPECT_FALSE(ofi.imbalance().has_value());  // one-sided
  ofi.reset();
  EXPECT_EQ(ofi.total(), 0);
  EXPECT_EQ(ofi.window_sum(), 0);
  EXPECT_TRUE(ofi.records().empty());
  book.submit({6, S, OrderType::Limit, Price{101}, 45, 6});  // ask from empty after reset: -45
  EXPECT_EQ(ofi.total(), -45);
  EXPECT_DOUBLE_EQ(*ofi.imbalance(), (55.0 - 45.0) / 100.0);
}

TEST(Analytics, SpreadIsChargedForHowLongItLasted) {
  SpreadDistribution spread(1);
  OrderBook book(&spread);
  book.submit({1, B, OrderType::Limit, Price{99}, 10, 10});   // one-sided from t=10
  book.submit({2, S, OrderType::Limit, Price{101}, 10, 12});  // spread 2 from t=12
  book.submit({3, S, OrderType::Limit, Price{100}, 10, 20});  // spread 1 from t=20
  book.submit({4, B, OrderType::Limit, Price{98}, 10, 20});   // same timestamp: deeper, no change
  book.cancel(3, 35);                                          // back to 2 from t=35
  using H = std::vector<std::pair<std::int64_t, std::uint64_t>>;
  EXPECT_EQ(spread.histogram(40), (H{{1, 15}, {2, 8 + 5}}));
  EXPECT_EQ(spread.one_sided_ns(40), 2u);
  EXPECT_EQ(spread.two_sided_ns(40), 28u);
  EXPECT_EQ(spread.elapsed(40), 30u);
  EXPECT_DOUBLE_EQ(spread.mean(40), (1.0 * 15 + 2.0 * 13) / 28.0);
  EXPECT_EQ(spread.spread(), 2);
  spread.reset(40);
  EXPECT_EQ(spread.histogram(50), (H{{2, 10}}));
  // A passing state inside one timestamp is charged nothing: lock and unlock at t=60.
  book.submit({5, S, OrderType::Limit, Price{100}, 1, 60});
  book.submit({6, B, OrderType::IOC, Price{100}, 1, 60});
  EXPECT_EQ(spread.histogram(70), (H{{2, 30}}));
}

TEST(Analytics, DepthProfileByTicksFromBest) {
  DepthProfile depth(/*tick=*/100, /*ticks=*/3);
  ArrayOrderBook book(&depth, 64, 100);
  book.submit({1, B, OrderType::Limit, Price{10'000}, 5, 0});
  book.submit({2, B, OrderType::Limit, Price{9'800}, 7, 0});  // 2 ticks from the best
  book.submit({3, B, OrderType::Limit, Price{9'500}, 9, 0});  // 5 ticks: outside the profile
  book.submit({4, S, OrderType::Limit, Price{10'100}, 4, 0});
  EXPECT_EQ(depth.current(B), (std::vector<Quantity>{5, 0, 7}));
  EXPECT_EQ(depth.current(S), (std::vector<Quantity>{4, 0, 0}));
  book.cancel(1, 10);  // best bid moves to 9,800: 9,500 is now 3 ticks away, still outside
  EXPECT_EQ(depth.current(B), (std::vector<Quantity>{7, 0, 0}));
  EXPECT_EQ(depth.sums(B, 30), (std::vector<std::uint64_t>{5 * 10 + 7 * 20, 0, 7 * 10}));
  EXPECT_EQ(depth.observed_ns(B, 30), 30u);
  EXPECT_DOUBLE_EQ(depth.mean(B, 30)[0], (50.0 + 140.0) / 30.0);
}

TEST(Analytics, QueuePositionsFollowPriorityRules) {
  QueuePositions q;
  OrderBook book(&q);
  for (OrderId id = 1; id <= 4; ++id) book.submit({id, S, OrderType::Limit, Price{101}, 10, id});
  q.track(3);
  EXPECT_EQ(q.ahead(3), (QueuePosition{2, 20}));
  book.modify(1, Price{101}, 4, 10);                // size cut in place: keeps its place
  EXPECT_EQ(q.ahead(3), (QueuePosition{2, 14}));
  book.modify(2, Price{101}, 15, 11);               // size up: to the back
  EXPECT_EQ(q.ahead(3), (QueuePosition{1, 4}));
  book.submit({9, B, OrderType::IOC, Price{101}, 6, 12});  // fills #1 (4) and 2 of #3
  EXPECT_EQ(q.ahead(3), (QueuePosition{0, 0}));
  EXPECT_FALSE(q.ahead(1).has_value());
  book.cancel(3, 20);
  q.flush();
  const auto& h = q.history();
  ASSERT_EQ(h.size(), 5u);
  EXPECT_EQ(h[0].ahead, (QueuePosition{2, 20}));  // when tracking began (t=4)
  EXPECT_EQ(h[0].ts, 4u);
  EXPECT_EQ(h[1].ahead, (QueuePosition{2, 14}));
  EXPECT_EQ(h[1].ts, 10u);
  EXPECT_EQ(h[2].ahead, (QueuePosition{1, 4}));
  EXPECT_EQ(h[3].ahead, (QueuePosition{0, 0}));
  EXPECT_EQ(h[3].leaves, 8);
  EXPECT_EQ(h[4].status, QueueStatus::Cancelled);
  EXPECT_EQ(h[4].ts, 20u);
  // A tracked id that arrives later is followed from its New report.
  q.track(50);
  book.submit({50, S, OrderType::Limit, Price{101}, 1, 30});
  book.submit({51, B, OrderType::IOC, Price{101}, 100, 40});
  q.flush();
  EXPECT_EQ(q.history().back().status, QueueStatus::Filled);
}

// ---------------------------------------------------------------- differential

// The best price and size of one side straight from the book's views.
std::optional<LevelView> best_of(const std::vector<LevelView>& d) {
  if (d.empty()) return std::nullopt;
  return d.front();
}

// Naive quote recorder: its own L2 (one hash map per side, best found by scanning) and OFI
// written out case by case, as the paper describes them.
class NaiveQuotes final : public Listener {
 public:
  void on_level_update(const LevelUpdate& u) override {
    auto& m = u.side == B ? bids : asks;
    if (u.qty == 0) m.erase(u.price.ticks);
    else m[u.price.ticks] = u.qty;
    const Quote q = current();
    if (q == prev) return;
    std::int64_t e = 0;
    // Bid side.
    if (!prev.bid && q.bid) e += q.bid_qty;
    else if (prev.bid && !q.bid) e -= prev.bid_qty;
    else if (prev.bid && q.bid) {
      if (*q.bid > *prev.bid) e += q.bid_qty;
      else if (*q.bid == *prev.bid) e += q.bid_qty - prev.bid_qty;
      else e -= prev.bid_qty;
    }
    // Ask side.
    if (!prev.ask && q.ask) e -= q.ask_qty;
    else if (prev.ask && !q.ask) e += prev.ask_qty;
    else if (prev.ask && q.ask) {
      if (*q.ask < *prev.ask) e -= q.ask_qty;
      else if (*q.ask == *prev.ask) e -= q.ask_qty - prev.ask_qty;
      else e += prev.ask_qty;
    }
    es.push_back(e);
    ts.push_back(u.ts);
    quotes.push_back(q);
    prev = q;
  }
  [[nodiscard]] Quote current() const {
    Quote q;
    for (const auto& [px, qty] : bids) {
      if (!q.bid || px > q.bid->ticks) q.bid = Price{px}, q.bid_qty = qty;
    }
    for (const auto& [px, qty] : asks) {
      if (!q.ask || px < q.ask->ticks) q.ask = Price{px}, q.ask_qty = qty;
    }
    return q;
  }
  std::unordered_map<std::int64_t, Quantity> bids, asks;
  Quote prev;
  std::vector<std::int64_t> es;
  std::vector<Timestamp> ts;
  std::vector<Quote> quotes;
};

template <typename Book>
BookState state_of(const Book& b) {
  return BookState{b.orders(B), b.orders(S)};
}

template <typename Book>
void apply_call(Book& b, const Call& c) {
  switch (c.kind) {
    case Op::Kind::Submit: b.submit(c.order); return;
    case Op::Kind::Cancel: b.cancel(c.order.id, c.order.ts); return;
    case Op::Kind::Modify: b.modify(c.order.id, c.order.price, c.order.qty, c.order.ts); return;
  }
}

struct BruteForceSample {
  Timestamp ts;
  OrderId id;
  QueuePosition ahead;
  Quantity leaves;
  QueueStatus status;
};

template <typename Book>
class AnalyticsDifferential : public ::testing::Test {};
using Books = ::testing::Types<OrderBook, ArrayOrderBook>;
struct BookNames {
  template <typename T>
  static std::string GetName(int) {
    return std::is_same_v<T, OrderBook> ? "MapLevels" : "ArrayLevels";
  }
};
TYPED_TEST_SUITE(AnalyticsDifferential, Books, BookNames);

TYPED_TEST(AnalyticsDifferential, EveryStatisticMatchesBruteForceOnEveryCall) {
  constexpr std::size_t kTicks = 8;
  constexpr std::size_t kWindow = 16;
  constexpr std::size_t kCalls = 3000;
  std::size_t checked_positions = 0;
  std::size_t tracked_samples = 0;
  for (std::size_t r = 0; r < kRegimes.size(); ++r) {
    for (std::uint64_t seed = 1; seed <= 2; ++seed) {
      SCOPED_TRACE(std::string(kRegimes[r].name) + " seed " + std::to_string(seed));
      SpreadDistribution spread(1);
      DepthProfile depth(1, kTicks);
      OrderFlowImbalance ofi(kWindow, /*record=*/true);
      QueuePositions queue;
      NaiveQuotes naive;
      EventDigest events;
      Fanout fan({&spread, &depth, &ofi, &queue, &naive, &events});
      TypeParam book(&fan, 4096, 1);

      std::map<std::int64_t, std::uint64_t> spread_bf;
      std::uint64_t one_sided_bf = 0;
      std::vector<std::uint64_t> depth_bf[2]{std::vector<std::uint64_t>(kTicks),
                                             std::vector<std::uint64_t>(kTicks)};
      std::uint64_t observed_bf[2]{0, 0};
      std::unordered_map<OrderId, BruteForceSample> last;  // tracked ids still resting
      std::vector<BruteForceSample> history_bf;

      const auto ops = generate_flow(kRegimes[r], seed, kCalls);
      Timestamp ts = 0;
      for (const Op& op : ops) {
        ++ts;
        const Call call = resolve(op, state_of(book), ts);
        apply_call(book, call);
        const BookState st = state_of(book);

        // Mirrors equal the book's own views.
        for (const Side s : {B, S}) {
          const auto d = book.depth(s, 1'000'000);
          const auto& mirror = spread.book().side(s);
          ASSERT_EQ(mirror.size(), d.size());
          for (const LevelView& lv : d) {
            const auto it = mirror.find(lv.price.ticks);
            ASSERT_NE(it, mirror.end());
            ASSERT_EQ(it->second.qty, lv.qty);
            ASSERT_EQ(it->second.count, lv.count);
          }
          const auto l3 = queue.orders(s);
          const auto& want = st.side(s);
          ASSERT_EQ(l3.size(), want.size());
          for (std::size_t i = 0; i < want.size(); ++i) {
            ASSERT_EQ(l3[i].id, want[i].id);
            ASSERT_EQ(l3[i].price, want[i].price);
            ASSERT_EQ(l3[i].leaves, want[i].qty);
          }
        }
        const auto bb = best_of(book.depth(B, 1));
        const auto ba = best_of(book.depth(S, 1));
        Quote q;
        if (bb) q.bid = bb->price, q.bid_qty = bb->qty;
        if (ba) q.ask = ba->price, q.ask_qty = ba->qty;
        ASSERT_EQ(spread.quote(), q);
        ASSERT_EQ(ofi.quote(), q);

        // Each call's end state lasts until the next call: one time unit.
        if (events.events() > 0) {
          if (bb && ba) spread_bf[ba->price.ticks - bb->price.ticks] += 1;
          else one_sided_bf += 1;
          for (const Side s : {B, S}) {
            const auto& side = st.side(s);
            if (side.empty()) continue;
            observed_bf[s == B ? 0 : 1] += 1;
            const std::int64_t best = side.front().price.ticks;
            for (const OrderView& o : side) {
              const auto k = static_cast<std::size_t>(std::abs(o.price.ticks - best));
              if (k < kTicks) depth_bf[s == B ? 0 : 1][k] += static_cast<std::uint64_t>(o.qty);
            }
          }
        }

        // Queue positions of every resting order, every 25 calls.
        if (ts % 25 == 0) {
          for (const Side s : {B, S}) {
            const auto& side = st.side(s);
            for (std::size_t i = 0; i < side.size(); ++i) {
              QueuePosition want;
              for (std::size_t j = 0; j < i; ++j) {
                if (side[j].price == side[i].price) ++want.orders, want.qty += side[j].qty;
              }
              ASSERT_EQ(queue.ahead(side[i].id), want) << "order " << side[i].id;
              ++checked_positions;
            }
          }
        }

        // Tracked histories: start tracking every 7th resting order; record by brute force.
        for (const Side s : {B, S}) {
          const auto& side = st.side(s);
          for (std::size_t i = 0; i < side.size(); ++i) {
            const OrderView& o = side[i];
            QueuePosition pos;
            for (std::size_t j = 0; j < i; ++j) {
              if (side[j].price == o.price) ++pos.orders, pos.qty += side[j].qty;
            }
            auto it = last.find(o.id);
            if (it == last.end() && o.id % 7 == 0 && o.ts == ts) {
              queue.track(o.id);  // just rested: follow it from here
              last[o.id] = {ts, o.id, pos, o.qty, QueueStatus::Resting};
              history_bf.push_back(last[o.id]);
            } else if (it != last.end() &&
                       (!(it->second.ahead == pos) || it->second.leaves != o.qty)) {
              it->second = {ts, o.id, pos, o.qty, QueueStatus::Resting};
              history_bf.push_back(it->second);
            }
          }
        }
        for (auto it = last.begin(); it != last.end();) {
          if (find_in(st, it->first) == nullptr) {
            const bool cancelled = call.kind == Op::Kind::Cancel && call.order.id == it->first;
            history_bf.push_back({ts, it->first, {}, 0,
                                  cancelled ? QueueStatus::Cancelled : QueueStatus::Filled});
            it = last.erase(it);
          } else {
            ++it;
          }
        }
      }

      const Timestamp now = ts + 1;
      ASSERT_FALSE(spread.overflowed());
      ASSERT_FALSE(depth.overflowed());
      EXPECT_EQ(spread.histogram(now),
                (std::vector<std::pair<std::int64_t, std::uint64_t>>(spread_bf.begin(),
                                                                     spread_bf.end())));
      EXPECT_EQ(spread.one_sided_ns(now), one_sided_bf);
      for (const Side s : {B, S}) {
        EXPECT_EQ(depth.sums(s, now), depth_bf[s == B ? 0 : 1]);
        EXPECT_EQ(depth.observed_ns(s, now), observed_bf[s == B ? 0 : 1]);
      }

      // OFI: every quote update, the window and the total, against the naive recorder.
      ASSERT_EQ(ofi.records().size(), naive.es.size());
      for (std::size_t i = 0; i < naive.es.size(); ++i) {
        ASSERT_EQ(ofi.records()[i].e, naive.es[i]) << "quote update " << i;
        ASSERT_EQ(ofi.records()[i].ts, naive.ts[i]);
        ASSERT_EQ(ofi.records()[i].quote, naive.quotes[i]);
      }
      EXPECT_EQ(ofi.total(), std::accumulate(naive.es.begin(), naive.es.end(), std::int64_t{0}));
      const std::size_t n = naive.es.size();
      const std::size_t w = std::min(n, kWindow);
      EXPECT_EQ(ofi.window_sum(),
                std::accumulate(naive.es.end() - static_cast<std::ptrdiff_t>(w), naive.es.end(),
                                std::int64_t{0}));
      EXPECT_EQ(ofi.updates(), n);

      // Tracked histories. Exits are recorded as they happen and moves when time moves on, so
      // within one timestamp the two lists may differ in order: compare them sorted.
      queue.flush();
      auto h = queue.history();
      const auto by_time_id = [](const auto& a, const auto& b) {
        return std::tie(a.ts, a.id, a.status) < std::tie(b.ts, b.id, b.status);
      };
      std::stable_sort(h.begin(), h.end(), by_time_id);
      std::stable_sort(history_bf.begin(), history_bf.end(), by_time_id);
      ASSERT_EQ(h.size(), history_bf.size());
      for (std::size_t i = 0; i < h.size(); ++i) {
        ASSERT_EQ(h[i].id, history_bf[i].id) << "sample " << i;
        ASSERT_EQ(h[i].ts, history_bf[i].ts) << "sample " << i;
        ASSERT_EQ(h[i].ahead, history_bf[i].ahead) << "sample " << i;
        ASSERT_EQ(h[i].leaves, history_bf[i].leaves) << "sample " << i;
        ASSERT_EQ(h[i].status, history_bf[i].status) << "sample " << i;
      }
      tracked_samples += h.size();
    }
  }
  EXPECT_GT(checked_positions, 10'000u);
  EXPECT_GT(tracked_samples, 1'000u);
}

}  // namespace
}  // namespace lob::testing_helpers
