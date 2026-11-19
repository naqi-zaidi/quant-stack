#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "lob/object_pool.hpp"

namespace {

struct Item {
  int a{0};
  double b{0.0};
};

TEST(ObjectPool, ReusesReleasedSlotsWithoutGrowing) {
  lob::ObjectPool<Item> pool(4);
  Item* p = pool.allocate();
  p->a = 7;
  pool.release(p);
  Item* q = pool.allocate();
  EXPECT_EQ(q, p);       // LIFO free list: the warm slot comes straight back
  EXPECT_EQ(q->a, 0);    // and it is handed out value-initialised
  EXPECT_EQ(pool.capacity(), 4u);
  EXPECT_EQ(pool.in_use(), 1u);
}

TEST(ObjectPool, GrowsWithoutMovingLiveObjects) {
  lob::ObjectPool<Item> pool(2);
  std::vector<Item*> live;
  for (int i = 0; i < 100; ++i) {
    live.push_back(pool.allocate());
    live.back()->a = i;
  }
  EXPECT_GE(pool.capacity(), 100u);
  EXPECT_EQ(pool.in_use(), 100u);
  for (int i = 0; i < 100; ++i) EXPECT_EQ(live[static_cast<std::size_t>(i)]->a, i);
  EXPECT_EQ(std::set<Item*>(live.begin(), live.end()).size(), 100u);  // all distinct
  for (Item* p : live) pool.release(p);
  EXPECT_EQ(pool.in_use(), 0u);
}

TEST(ObjectPool, EnsureFreeMakesLaterAllocationsHeapFree) {
  lob::ObjectPool<Item> pool(1);
  pool.ensure_free(50);
  const auto cap = pool.capacity();
  for (int i = 0; i < 50; ++i) (void)pool.allocate();
  EXPECT_EQ(pool.capacity(), cap);
}

}  // namespace
