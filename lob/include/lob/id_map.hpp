#pragma once

// Open-addressing hash map from order id to a pointer, for O(1) cancel and modify.
//
// Linear probing over a power-of-two table kept at most half full, with backward-shift
// deletion: erasing a key slides later members of its probe chain back, so there are no
// tombstones and lookups never slow down as orders churn. A null value marks an empty slot,
// so every id (including 0) is a valid key. Inserts allocate only when the table doubles,
// which reserve() moves off the hot path.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lob/types.hpp"

namespace lob {

template <typename T>
class IdMap {
 public:
  explicit IdMap(std::size_t expected = 1024) { rehash(table_size_for(expected)); }

  [[nodiscard]] T* find(OrderId key) const noexcept {
    for (std::size_t i = home(key);; i = (i + 1) & mask_) {
      const Slot& s = slots_[i];
      if (s.value == nullptr) return nullptr;
      if (s.key == key) return s.value;
    }
  }

  // Inserts key -> value (value must be non-null). Returns false, changing nothing, if the
  // key is already present.
  bool insert(OrderId key, T* value) {
    reserve(size_ + 1);
    std::size_t i = home(key);
    for (; slots_[i].value != nullptr; i = (i + 1) & mask_) {
      if (slots_[i].key == key) return false;
    }
    slots_[i] = Slot{key, value};
    ++size_;
    return true;
  }

  // Removes key. Returns false if it was not present.
  bool erase(OrderId key) noexcept {
    std::size_t i = home(key);
    for (;; i = (i + 1) & mask_) {
      if (slots_[i].value == nullptr) return false;
      if (slots_[i].key == key) break;
    }
    // Backward shift: walk the rest of the chain and pull back any entry whose home slot
    // is not cyclically inside (hole, j]; such an entry would become unreachable otherwise.
    std::size_t hole = i;
    for (std::size_t j = (i + 1) & mask_; slots_[j].value != nullptr; j = (j + 1) & mask_) {
      const std::size_t h = home(slots_[j].key);
      const bool stays = hole <= j ? (hole < h && h <= j) : (hole < h || h <= j);
      if (!stays) {
        slots_[hole] = slots_[j];
        hole = j;
      }
    }
    slots_[hole] = Slot{};
    --size_;
    return true;
  }

  // Guarantees that the map can hold `n` keys without rehashing.
  void reserve(std::size_t n) {
    if (n > slots_.size() / 2) rehash(table_size_for(n));
  }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t table_size() const noexcept { return slots_.size(); }

 private:
  struct Slot {
    OrderId key{0};
    T* value{nullptr};
  };

  // splitmix64 finaliser. Exchange ids are often sequential; mixing spreads them so probe
  // chains stay short whatever the id pattern.
  [[nodiscard]] static constexpr std::uint64_t mix(std::uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  }

  [[nodiscard]] std::size_t home(OrderId key) const noexcept {
    return static_cast<std::size_t>(mix(key)) & mask_;
  }

  [[nodiscard]] static std::size_t table_size_for(std::size_t n) {
    return std::bit_ceil(n < 8 ? std::size_t{16} : 2 * n);
  }

  void rehash(std::size_t new_size) {
    std::vector<Slot> old(new_size);
    old.swap(slots_);
    mask_ = slots_.size() - 1;
    for (const Slot& s : old) {
      if (s.value == nullptr) continue;
      std::size_t i = home(s.key);
      while (slots_[i].value != nullptr) i = (i + 1) & mask_;
      slots_[i] = s;
    }
  }

  std::vector<Slot> slots_;
  std::size_t mask_{0};
  std::size_t size_{0};
};

}  // namespace lob
