// Phase 2 step 4 gate: heap allocations on the engine's hot path.
//
// Replaces global operator new/delete with counting versions, and counts only while an
// engine call (submit, cancel, modify) is running. Workloads run after a warm-up, so
// one-off growth (pool, id map, array pages) is excluded and what remains is steady state:
//   - the five property-test regimes, several fixed-seed flows each (always);
//   - optionally a real LOBSTER day, every engine call the replay makes (local data).
//
// Gate, per workload:
//   array book  zero allocations and zero frees inside engine calls;
//   map book    exactly one allocation per price level opened and one free per level closed
//               (the std::map node), and nothing else.
// Allocation counts are deterministic, unlike latency on shared CI runners, so this is the
// step's CI gate. Run it in Release only: sanitizers replace the allocator.
//
//   lob_alloc_check [LOBSTER message file]

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <new>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lob/lobster.hpp"
#include "lob/order_book.hpp"
#include "property_harness.hpp"

// ---------------------------------------------------------------- counting allocator

namespace alloc_count {
bool counting = false;
std::uint64_t allocs = 0;
std::uint64_t frees = 0;

void* allocate(std::size_t n) {
  if (counting) ++allocs;
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void* allocate_aligned(std::size_t n, std::align_val_t a) {
  if (counting) ++allocs;
  const auto al = static_cast<std::size_t>(a);
  const std::size_t size = (n + al - 1) / al * al;
#ifdef _MSC_VER
  if (void* p = _aligned_malloc(size == 0 ? al : size, al)) return p;
#else
  if (void* p = std::aligned_alloc(al, size == 0 ? al : size)) return p;
#endif
  throw std::bad_alloc();
}
void release(void* p) noexcept {
  if (p != nullptr && counting) ++frees;
  std::free(p);
}
void release_aligned(void* p) noexcept {
  if (p != nullptr && counting) ++frees;
#ifdef _MSC_VER
  _aligned_free(p);
#else
  std::free(p);
#endif
}

// Turns counting off for its lifetime (the measuring code's own bookkeeping).
class Paused {
 public:
  Paused() : was_(counting) { counting = false; }
  Paused(const Paused&) = delete;
  Paused& operator=(const Paused&) = delete;
  ~Paused() { counting = was_; }
  [[nodiscard]] bool was_counting() const noexcept { return was_; }

 private:
  bool was_;
};
}  // namespace alloc_count

void* operator new(std::size_t n) { return alloc_count::allocate(n); }
void* operator new[](std::size_t n) { return alloc_count::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return alloc_count::allocate_aligned(n, a); }
void* operator new[](std::size_t n, std::align_val_t a) {
  return alloc_count::allocate_aligned(n, a);
}
void operator delete(void* p) noexcept { alloc_count::release(p); }
void operator delete[](void* p) noexcept { alloc_count::release(p); }
void operator delete(void* p, std::size_t) noexcept { alloc_count::release(p); }
void operator delete[](void* p, std::size_t) noexcept { alloc_count::release(p); }
void operator delete(void* p, std::align_val_t) noexcept { alloc_count::release_aligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { alloc_count::release_aligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
  alloc_count::release_aligned(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
  alloc_count::release_aligned(p);
}

// ---------------------------------------------------------------- measurement

namespace {

using namespace lob;
using namespace lob::testing_helpers;

// Counts price levels opened and closed during counted engine calls, from the book's level
// updates. Its own set operations run with counting paused.
class LevelCounter final : public Listener {
 public:
  void on_level_update(const LevelUpdate& u) override {
    const alloc_count::Paused pause;
    const auto key = std::make_pair(u.side == Side::Buy ? 0 : 1, u.price.ticks);
    bool changed = false;
    if (u.qty > 0) {
      changed = open_.insert(key).second;
      if (changed && pause.was_counting()) ++opened;
    } else {
      changed = open_.erase(key) == 1;
      if (changed && pause.was_counting()) ++closed;
    }
  }
  std::uint64_t opened{0};
  std::uint64_t closed{0};

 private:
  std::set<std::pair<int, std::int64_t>> open_;
};

struct Counts {
  std::string workload;
  std::string book;
  std::uint64_t calls{0}, allocs{0}, frees{0}, opened{0}, closed{0};
  bool array{false};

  [[nodiscard]] bool ok() const {
    return calls > 0 && (array ? allocs == 0 && frees == 0 : allocs == opened && frees == closed);
  }
};

template <typename Fn>
void counted(Fn&& f) {
  alloc_count::counting = true;
  f();
  alloc_count::counting = false;
}

// One book, warm-up flow first, then fixed-seed flows of every regime run back to back.
// Order ids are offset per flow so later flows never collide with orders still resting.
template <typename Book>
Counts synthetic(const char* name, bool is_array, std::uint64_t flows_per_regime) {
  LevelCounter levels;
  Book book(&levels, 1 << 16);
  Counts c{"synthetic: 5 regimes x " + std::to_string(flows_per_regime) + " flows x 2,000 calls",
           name};
  c.array = is_array;
  auto run = [&](const std::vector<Op>& ops, OrderId offset, bool count) {
    for (std::size_t i = 0; i < ops.size(); ++i) {
      Op op = ops[i];
      op.order.id += offset;
      op.target += offset;
      const BookState st{book.orders(Side::Buy), book.orders(Side::Sell)};
      const Call call = resolve(op, st, static_cast<Timestamp>(i));
      auto engine = [&] {
        switch (call.kind) {
          case Op::Kind::Submit: book.submit(call.order); break;
          case Op::Kind::Cancel: book.cancel(call.order.id, call.order.ts); break;
          case Op::Kind::Modify:
            book.modify(call.order.id, call.order.price, call.order.qty, call.order.ts);
            break;
        }
      };
      if (count) {
        counted(engine);
        ++c.calls;
      } else {
        engine();
      }
    }
  };
  for (const Regime& rg : kRegimes) run(generate_flow(rg, 9999, 2000), 0, false);  // warm-up
  OrderId offset = 0;
  const std::uint64_t a0 = alloc_count::allocs, f0 = alloc_count::frees;
  const std::uint64_t o0 = levels.opened, c0 = levels.closed;
  for (std::uint64_t seed = 1; seed <= flows_per_regime; ++seed) {
    for (const Regime& rg : kRegimes) run(generate_flow(rg, seed, 2000), offset += 10'000'000, true);
  }
  c.allocs = alloc_count::allocs - a0;
  c.frees = alloc_count::frees - f0;
  c.opened = levels.opened - o0;
  c.closed = levels.closed - c0;
  return c;
}

// Brackets every engine call the replay makes.
class CountingObserver final : public lobster::CallObserver {
 public:
  void begin(Kind) override {
    alloc_count::counting = true;
  }
  void end(Kind) override {
    alloc_count::counting = false;
    ++calls;
  }
  std::uint64_t calls{0};
};

template <typename R>
Counts real_flow(const std::string& path, const char* name, bool is_array, std::int64_t tick) {
  const lobster::FileInfo info = lobster::describe_file(path);
  lobster::DayReader rd(path, info.orderbook_path, info.levels);
  lobster::Message m;
  lobster::Snapshot s;
  if (!rd.next(m, s)) throw std::runtime_error("empty file");
  R rp(info.levels, tick);
  LevelCounter levels;
  CountingObserver obs;
  rp.set_listener(&levels);
  rp.set_call_observer(&obs);
  rp.seed(s, m.ns);
  const std::uint64_t a0 = alloc_count::allocs, f0 = alloc_count::frees;
  while (rd.next(m, s)) rp.step(m, s);
  if (rp.stats().mismatches != 0 || rp.stats().errors != 0) {
    throw std::runtime_error("replay did not match LOBSTER; run lob_replay for details");
  }
  Counts c{info.ticker + " " + info.date + " (real flow, every engine call)", name};
  c.array = is_array;
  c.calls = obs.calls;
  c.allocs = alloc_count::allocs - a0;
  c.frees = alloc_count::frees - f0;
  c.opened = levels.opened;
  c.closed = levels.closed;
  return c;
}

}  // namespace

int main(int argc, char** argv) try {
  std::vector<Counts> rows;
  rows.push_back(synthetic<OrderBook>("map", false, 8));
  rows.push_back(synthetic<ArrayOrderBook>("array", true, 8));
  if (argc > 1) {
    rows.push_back(real_flow<lobster::Replay>(argv[1], "map", false, 1));
    rows.push_back(real_flow<lobster::ArrayReplay>(argv[1], "array", true, 100));
  }

  bool ok = true;
  std::cout << "## Heap allocations inside engine calls, steady state\n\n"
            << "| Workload | Book | Engine calls | Allocations | Frees | Levels opened | "
               "Levels closed | Allocations per call | Gate |\n"
            << "|---|---|---:|---:|---:|---:|---:|---:|:---:|\n";
  for (const Counts& c : rows) {
    ok = ok && c.ok();
    std::cout << "| " << c.workload << " | " << c.book << " | " << c.calls << " | " << c.allocs
              << " | " << c.frees << " | " << c.opened << " | " << c.closed << " | "
              << std::fixed << std::setprecision(4)
              << static_cast<double>(c.allocs) / static_cast<double>(c.calls) << " | "
              << (c.ok() ? "pass" : "**FAIL**") << " |\n";
  }
  std::cout << "\nGate: the array book allocates and frees nothing; the map book allocates one "
               "node per level opened and frees one per level closed, and nothing else.\n"
            << (ok ? "PASS" : "FAIL") << "\n";
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& e) {
  std::cerr << "lob_alloc_check: " << e.what() << "\n";
  return 2;
}
