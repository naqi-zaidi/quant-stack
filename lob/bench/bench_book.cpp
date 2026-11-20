// Phase 2 step 4: microbenchmarks of single order-book operations, map vs array levels.
//
// Every benchmark runs against the same background book: 50 price levels a side around a
// $580 mid on a one-cent grid (LOBSTER units: tick 100), 20 resting orders per level. Each
// timed batch performs K = 1024 operations of one kind; the setup and cleanup that keep the
// background book identical between batches run with the timer paused. No listener is
// attached, so this is the engine alone.
//
//   lob_bench --benchmark_repetitions=10 --benchmark_out=bench.json --benchmark_out_format=json
//
// Reported time is CPU time per batch; the per_op counter divides by the operations in it.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "lob/order_book.hpp"

namespace {

using lob::ArrayOrderBook;
using lob::Order;
using lob::OrderBook;
using lob::OrderId;
using lob::OrderType;
using lob::Price;
using lob::Side;

constexpr std::int64_t kTick = 100;         // one cent
constexpr std::int64_t kMid = 5'800'000;    // $580.00
constexpr std::int64_t kLevels = 50;        // per side
constexpr int kPerLevel = 20;
constexpr std::size_t kBatch = 1024;
constexpr OrderId kFirstId = 1'000'000;

// A book holding the background levels, plus helpers for one batch of operations.
template <typename Book>
struct Bench {
  Book book{nullptr, 1 << 16, kTick};
  OrderId next_id{kFirstId};
  std::mt19937_64 rng{7};
  std::vector<OrderId> ids;
  std::vector<Price> prices;
  std::vector<Side> sides;

  // `gap`: extra empty ticks between the mid and the background levels on each side.
  explicit Bench(std::int64_t gap = 0) {
    OrderId id = 1;
    for (std::int64_t l = 1; l <= kLevels; ++l) {
      for (int k = 0; k < kPerLevel; ++k) {
        book.submit(Order{id++, Side::Buy, OrderType::Limit, Price{kMid - (gap + l) * kTick}, 100});
        book.submit(Order{id++, Side::Sell, OrderType::Limit, Price{kMid + (gap + l) * kTick}, 100});
      }
    }
    ids.resize(kBatch);
    prices.resize(kBatch);
    sides.resize(kBatch);
  }

  // Random resting prices at existing levels (no level is opened).
  void pick_existing_levels() {
    std::uniform_int_distribution<std::int64_t> lvl(1, kLevels);
    for (std::size_t i = 0; i < kBatch; ++i) {
      sides[i] = (i % 2 == 0) ? Side::Buy : Side::Sell;
      const std::int64_t off = lvl(rng) * kTick;
      prices[i] = Price{sides[i] == Side::Buy ? kMid - off : kMid + off};
    }
  }
  // Fresh prices beyond the background book: every order opens its own level.
  void pick_new_levels() {
    for (std::size_t i = 0; i < kBatch; ++i) {
      sides[i] = (i % 2 == 0) ? Side::Buy : Side::Sell;
      const auto off = (kLevels + 1 + static_cast<std::int64_t>(i / 2)) * kTick;
      prices[i] = Price{sides[i] == Side::Buy ? kMid - off : kMid + off};
    }
  }
  void add_batch() {
    for (std::size_t i = 0; i < kBatch; ++i) {
      ids[i] = next_id++;
      book.submit(Order{ids[i], sides[i], OrderType::Limit, prices[i], 100});
    }
  }
  void cancel_batch() {
    for (OrderId id : ids) book.cancel(id);
  }
};

void report(benchmark::State& state) {
  const std::int64_t ops = state.iterations() * static_cast<std::int64_t>(kBatch);
  state.SetItemsProcessed(ops);
  state.counters["per_op"] = benchmark::Counter(
      static_cast<double>(ops),
      benchmark::Counter::kIsRate | benchmark::Counter::kInvert);  // seconds per operation
}

// Add a resting limit order at an existing level.
template <typename Book>
void BM_Add(benchmark::State& state) {
  Bench<Book> b;
  for (auto _ : state) {
    state.PauseTiming();
    b.pick_existing_levels();
    state.ResumeTiming();
    b.add_batch();
    state.PauseTiming();
    b.cancel_batch();
    state.ResumeTiming();
  }
  report(state);
}

// Add a resting limit order that opens a new price level.
template <typename Book>
void BM_AddNewLevel(benchmark::State& state) {
  Bench<Book> b;
  b.pick_new_levels();
  for (auto _ : state) {
    b.add_batch();
    state.PauseTiming();
    b.cancel_batch();
    state.ResumeTiming();
  }
  report(state);
}

// Cancel a resting order from inside a queue (the level survives).
template <typename Book>
void BM_Cancel(benchmark::State& state) {
  Bench<Book> b;
  for (auto _ : state) {
    state.PauseTiming();
    b.pick_existing_levels();
    b.add_batch();
    std::shuffle(b.ids.begin(), b.ids.end(), b.rng);
    state.ResumeTiming();
    b.cancel_batch();
  }
  report(state);
}

// Cancel the only order at a level, so the level closes.
template <typename Book>
void BM_CancelClosesLevel(benchmark::State& state) {
  Bench<Book> b;
  b.pick_new_levels();
  for (auto _ : state) {
    state.PauseTiming();
    b.add_batch();
    std::shuffle(b.ids.begin(), b.ids.end(), b.rng);
    state.ResumeTiming();
    b.cancel_batch();
  }
  report(state);
}

// Reduce an order's size in place (keeps queue priority).
template <typename Book>
void BM_ModifyDown(benchmark::State& state) {
  Bench<Book> b;
  for (auto _ : state) {
    state.PauseTiming();
    b.pick_existing_levels();
    b.add_batch();
    state.ResumeTiming();
    for (std::size_t i = 0; i < kBatch; ++i) b.book.modify(b.ids[i], b.prices[i], 50);
    state.PauseTiming();
    b.cancel_batch();
    state.ResumeTiming();
  }
  report(state);
}

// An IOC that fills exactly one resting order at the best price. The K resting orders sit
// alone at a new best ask, so the last fill also closes the level and moves the best price.
template <typename Book>
void BM_MatchOne(benchmark::State& state) {
  Bench<Book> b;
  for (auto _ : state) {
    state.PauseTiming();
    for (std::size_t i = 0; i < kBatch; ++i) {
      b.ids[i] = b.next_id++;
      b.book.submit(Order{b.ids[i], Side::Sell, OrderType::Limit, Price{kMid}, 100});
    }
    state.ResumeTiming();
    for (std::size_t i = 0; i < kBatch; ++i) {
      b.book.submit(Order{b.next_id++, Side::Buy, OrderType::IOC, Price{kMid}, 100});
    }
  }
  report(state);
}

// An aggressive order that sweeps five price levels (100 orders) and closes all five. The
// background book leaves a wide gap each side of the mid; before each batch, 64 x 5 fresh
// ask levels are built inside it, and the timed batch is 64 sweeps, each taking the five
// best levels. The background is never touched.
constexpr std::int64_t kSweeps = 64;

template <typename Book>
void BM_Sweep5Levels(benchmark::State& state) {
  Bench<Book> b(5 * kSweeps);
  for (auto _ : state) {
    state.PauseTiming();
    for (std::int64_t l = 0; l < 5 * kSweeps; ++l) {
      for (int k = 0; k < kPerLevel; ++k) {
        b.book.submit(Order{b.next_id++, Side::Sell, OrderType::Limit,
                            Price{kMid - (5 * kSweeps - 1 - l) * kTick}, 100});
      }
    }
    state.ResumeTiming();
    for (std::int64_t sweep = 0; sweep < kSweeps; ++sweep) {
      const Price limit{kMid - (5 * kSweeps - 5 - 5 * sweep) * kTick};  // the 5th level up
      b.book.submit(Order{b.next_id++, Side::Buy, OrderType::IOC, limit, 5 * kPerLevel * 100});
    }
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * kSweeps));
  state.counters["per_op"] = benchmark::Counter(static_cast<double>(state.iterations() * kSweeps),
                                                benchmark::Counter::kIsRate |
                                                    benchmark::Counter::kInvert);
}

}  // namespace

#define LOB_BENCH(fn)                                                         \
  BENCHMARK_TEMPLATE(fn, OrderBook)->Name(#fn "/map");                        \
  BENCHMARK_TEMPLATE(fn, ArrayOrderBook)->Name(#fn "/array")

LOB_BENCH(BM_Add);
LOB_BENCH(BM_AddNewLevel);
LOB_BENCH(BM_Cancel);
LOB_BENCH(BM_CancelClosesLevel);
LOB_BENCH(BM_ModifyDown);
LOB_BENCH(BM_MatchOne);
LOB_BENCH(BM_Sweep5Levels);

BENCHMARK_MAIN();
