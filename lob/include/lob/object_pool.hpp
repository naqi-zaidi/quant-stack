#pragma once

// Fixed-address object pool. Objects live in chunks that are never moved or freed while the
// pool exists, so raw pointers into the pool stay valid for an object's whole lifetime.
// Steady state (allocate after release) touches only the free list: no heap traffic. The
// heap is hit only when the pool runs dry and grows by a new chunk, which callers avoid by
// sizing it up front.

#include <cstddef>
#include <memory>
#include <vector>

namespace lob {

template <typename T>
class ObjectPool {
 public:
  explicit ObjectPool(std::size_t initial_capacity = 1024) { grow(initial_capacity); }

  ObjectPool(const ObjectPool&) = delete;
  ObjectPool& operator=(const ObjectPool&) = delete;
  ObjectPool(ObjectPool&&) noexcept = default;
  ObjectPool& operator=(ObjectPool&&) noexcept = default;
  ~ObjectPool() = default;

  // Returns a value-initialised object. Grows (allocates) only when no free slot is left.
  [[nodiscard]] T* allocate() {
    ensure_free(1);
    T* p = free_.back();
    free_.pop_back();
    *p = T{};
    return p;
  }

  // Hands a slot back. The pointer must come from this pool and must not be used afterwards.
  void release(T* p) noexcept { free_.push_back(p); }  // capacity reserved in grow(): no throw

  // Makes sure the next `n` allocations will not touch the heap.
  void ensure_free(std::size_t n) {
    if (free_.size() < n) grow(n - free_.size() > capacity_ ? n - free_.size() : capacity_);
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t in_use() const noexcept { return capacity_ - free_.size(); }

 private:
  void grow(std::size_t n) {
    if (n == 0) n = 1;
    auto chunk = std::make_unique<T[]>(n);
    free_.reserve(capacity_ + n);  // so release() can never reallocate
    // Push in reverse so allocation order walks the chunk forwards (friendlier to the cache).
    for (std::size_t i = n; i > 0; --i) free_.push_back(&chunk[i - 1]);
    chunks_.push_back(std::move(chunk));
    capacity_ += n;
  }

  std::vector<std::unique_ptr<T[]>> chunks_;
  std::vector<T*> free_;
  std::size_t capacity_{0};
};

}  // namespace lob
