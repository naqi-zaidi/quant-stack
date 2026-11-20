#pragma once

// Price-level storage for one side of the book, in two interchangeable forms.
//
//   MapLevels    std::map from price to level: any price, O(log L) to open or close a level,
//                one heap node per level.
//   ArrayLevels  a tick-indexed array: price / tick is the slot. O(1) to reach any level and
//                no allocation once its page exists. Finding the next-best level after the
//                best one empties scans an occupancy bitmap, 64 slots per instruction. Slots
//                live in fixed 4096-slot pages that never move, so Level* stays valid.
//
// Both expose the same interface, so OrderBook is written once against it:
//   accepts(p)          p is a valid resting price here (positive, on the tick grid, in range)
//   empty() size()      levels currently open
//   best()              best open level (the side must not be empty)
//   get_or_create(p)    open level at p (reference stays valid until erase(p))
//   find(p)             open level at p, or nullptr
//   erase(p)            close the (already empty) level at p
//   walk(f)             f(const Level&) on open levels best-first while f returns true

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "lob/types.hpp"

namespace lob {

struct Level;

struct OrderNode {
  OrderId id{};
  Price price{};
  Quantity qty{};      // open (unfilled) quantity
  Timestamp ts{};      // event time of the message that gave the order its current priority
  std::uint64_t seq{}; // arrival sequence number; strictly increasing head -> tail in a level
  Side side{Side::Buy};
  OrderType type{OrderType::Limit};
  OrderNode* prev{nullptr};
  OrderNode* next{nullptr};
  Level* level{nullptr};
};

struct Level {
  Price price{};
  Quantity total{0};         // sum of open quantity of the orders below
  std::uint32_t count{0};
  OrderNode* head{nullptr};  // oldest: matched first
  OrderNode* tail{nullptr};  // newest
};

namespace detail {

// Priority order of one side. kWorse is the direction of worse prices in tick-index space.
struct HigherFirst {  // bids
  static constexpr std::int64_t kWorse = -1;
  constexpr bool operator()(Price a, Price b) const noexcept { return a > b; }
};
struct LowerFirst {  // asks
  static constexpr std::int64_t kWorse = +1;
  constexpr bool operator()(Price a, Price b) const noexcept { return a < b; }
};

template <typename BestFirst>
class MapSide {
 public:
  explicit MapSide(std::int64_t tick = 1) : tick_(tick) {}

  [[nodiscard]] bool accepts(Price p) const noexcept {
    return p.ticks > 0 && (tick_ == 1 || p.ticks % tick_ == 0);
  }
  [[nodiscard]] bool empty() const noexcept { return levels_.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return levels_.size(); }
  [[nodiscard]] Level& best() noexcept { return levels_.begin()->second; }
  [[nodiscard]] const Level& best() const noexcept { return levels_.begin()->second; }

  // Map nodes never move, so the reference stays valid until the level is erased.
  Level& get_or_create(Price price) {
    auto [it, inserted] = levels_.try_emplace(price);
    if (inserted) it->second.price = price;
    return it->second;
  }

  [[nodiscard]] const Level* find(Price price) const {
    auto it = levels_.find(price);
    return it == levels_.end() ? nullptr : &it->second;
  }

  void erase(Price price) { levels_.erase(price); }

  template <typename F>
  void walk(F&& f) const {
    for (const auto& kv : levels_) {
      if (!f(kv.second)) return;
    }
  }

 private:
  std::int64_t tick_;
  std::map<Price, Level, BestFirst> levels_;
};

template <typename BestFirst>
class ArraySide {
 public:
  static constexpr int kPageBits = 12;
  static constexpr std::int64_t kPageSize = std::int64_t{1} << kPageBits;  // 4096 slots
  static constexpr std::int64_t kWords = kPageSize / 64;
  static constexpr std::int64_t kMaxPages = 1 << 14;  // span limit: 2^26 ticks from low to high

  explicit ArraySide(std::int64_t tick = 1) : tick_(tick) {}

  // One modulo, like the map: the range check compares against price limits kept up to date
  // as pages are added, because a 64-bit division per order is measurable on the hot path.
  [[nodiscard]] bool accepts(Price p) const noexcept {
    return p.ticks > 0 && (tick_ == 1 || p.ticks % tick_ == 0) && p.ticks >= lo_limit_ &&
           p.ticks <= hi_limit_;
  }
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] Level& best() noexcept { return *slot(best_); }
  [[nodiscard]] const Level& best() const noexcept { return *slot(best_); }

  // Allocates (at most) a page and the page table entry, before touching any level state.
  Level& get_or_create(Price price) {
    const std::int64_t t = index_of(price);
    Page& page = page_for(t);
    const auto off = static_cast<std::size_t>(t & (kPageSize - 1));
    Level& l = page.levels[off];
    std::uint64_t& word = page.bits[off >> 6];
    const std::uint64_t bit = std::uint64_t{1} << (off & 63);
    if ((word & bit) == 0) {
      word |= bit;
      l = Level{};
      l.price = price;
      if (count_ == 0 || better(t, best_)) best_ = t;
      ++count_;
    }
    return l;
  }

  [[nodiscard]] const Level* find(Price price) const {
    if (price.ticks <= 0 || price.ticks % tick_ != 0) return nullptr;
    const std::int64_t t = index_of(price);
    return occupied(t) ? slot(t) : nullptr;
  }

  void erase(Price price) {
    const std::int64_t t = index_of(price);
    Page* page = page_ptr(page_of(t));
    const auto off = static_cast<std::size_t>(t & (kPageSize - 1));
    page->bits[off >> 6] &= ~(std::uint64_t{1} << (off & 63));
    page->levels[off] = Level{};
    --count_;
    if (count_ > 0 && t == best_) best_ = *next_occupied(t);  // something worse must exist
  }

  template <typename F>
  void walk(F&& f) const {
    if (count_ == 0) return;
    std::optional<std::int64_t> t = best_;
    for (std::size_t seen = 0; seen < count_ && t; ++seen, t = next_occupied(*t)) {
      if (!f(*slot(*t))) return;
    }
  }

 private:
  struct Page {
    std::array<Level, kPageSize> levels;
    std::array<std::uint64_t, kWords> bits{};
  };

  static constexpr bool better(std::int64_t a, std::int64_t b) noexcept {
    return BestFirst::kWorse < 0 ? a > b : a < b;
  }
  [[nodiscard]] std::int64_t index_of(Price p) const noexcept { return p.ticks / tick_; }
  static constexpr std::int64_t page_of(std::int64_t t) noexcept { return t >> kPageBits; }

  [[nodiscard]] Page* page_ptr(std::int64_t pg) const noexcept {
    const std::int64_t k = pg - first_page_;
    if (pages_.empty() || k < 0 || k >= static_cast<std::int64_t>(pages_.size())) return nullptr;
    return pages_[static_cast<std::size_t>(k)].get();
  }
  [[nodiscard]] Level* slot(std::int64_t t) const noexcept {
    return &page_ptr(page_of(t))->levels[static_cast<std::size_t>(t & (kPageSize - 1))];
  }
  [[nodiscard]] bool occupied(std::int64_t t) const noexcept {
    const Page* page = page_ptr(page_of(t));
    if (page == nullptr) return false;
    const auto off = static_cast<std::size_t>(t & (kPageSize - 1));
    return (page->bits[off >> 6] >> (off & 63)) & 1U;
  }

  Page& page_for(std::int64_t t) {
    const std::int64_t pg = page_of(t);
    if (pages_.empty()) {
      first_page_ = pg;
      pages_.emplace_back();
    } else if (pg < first_page_) {  // grow downward: shift page pointers, pages stay put
      const auto extra = static_cast<std::size_t>(first_page_ - pg);
      std::vector<std::unique_ptr<Page>> grown(extra + pages_.size());
      std::move(pages_.begin(), pages_.end(), grown.begin() + static_cast<std::ptrdiff_t>(extra));
      pages_.swap(grown);
      first_page_ = pg;
    } else if (const std::int64_t k = pg - first_page_; k >= static_cast<std::int64_t>(pages_.size())) {
      pages_.resize(static_cast<std::size_t>(k + 1));
    }
    auto& p = pages_[static_cast<std::size_t>(pg - first_page_)];
    if (!p) p = std::make_unique<Page>();
    // Prices reachable without the page table spanning more than kMaxPages pages.
    const std::int64_t last = first_page_ + static_cast<std::int64_t>(pages_.size()) - 1;
    lo_limit_ = std::max<std::int64_t>(1, ((last - kMaxPages + 1) << kPageBits) * tick_);
    hi_limit_ = (((first_page_ + kMaxPages) << kPageBits) - 1) * tick_;
    return *p;
  }

  // Next open slot strictly beyond `t` in the worse direction, or nullopt.
  [[nodiscard]] std::optional<std::int64_t> next_occupied(std::int64_t t) const noexcept {
    const std::int64_t lo = first_page_ << kPageBits;
    const std::int64_t hi = (first_page_ + static_cast<std::int64_t>(pages_.size())) << kPageBits;
    for (std::int64_t i = t + BestFirst::kWorse; i >= lo && i < hi;) {
      const std::int64_t pg = page_of(i);
      const Page* page = page_ptr(pg);
      const std::int64_t base = pg << kPageBits;
      if (page == nullptr) {  // skip a missing page in one step
        i = BestFirst::kWorse > 0 ? base + kPageSize : base - 1;
        continue;
      }
      const std::int64_t off = i - base;
      std::int64_t w = off >> 6;
      const int b = static_cast<int>(off & 63);
      if constexpr (BestFirst::kWorse > 0) {
        std::uint64_t bits = page->bits[static_cast<std::size_t>(w)] & (~std::uint64_t{0} << b);
        for (;;) {
          if (bits != 0) return base + w * 64 + std::countr_zero(bits);
          if (++w == kWords) break;
          bits = page->bits[static_cast<std::size_t>(w)];
        }
        i = base + kPageSize;
      } else {
        std::uint64_t bits = page->bits[static_cast<std::size_t>(w)] &
                             (b == 63 ? ~std::uint64_t{0} : (std::uint64_t{1} << (b + 1)) - 1);
        for (;;) {
          if (bits != 0) return base + w * 64 + 63 - std::countl_zero(bits);
          if (w-- == 0) break;
          bits = page->bits[static_cast<std::size_t>(w)];
        }
        i = base - 1;
      }
    }
    return std::nullopt;
  }

  std::int64_t tick_;
  std::vector<std::unique_ptr<Page>> pages_;  // pages_[k] holds slots of page first_page_ + k
  std::int64_t first_page_{0};
  std::size_t count_{0};
  std::int64_t best_{0};  // tick index of the best open level (valid when count_ > 0)
  std::int64_t lo_limit_{1};  // lowest and highest acceptable prices (see accepts)
  std::int64_t hi_limit_{std::int64_t{1} << 62};
};

}  // namespace detail

// Level-storage policies for BasicOrderBook.
struct MapLevels {
  template <typename BestFirst>
  using Side = detail::MapSide<BestFirst>;
};
struct ArrayLevels {
  template <typename BestFirst>
  using Side = detail::ArraySide<BestFirst>;
};

}  // namespace lob
