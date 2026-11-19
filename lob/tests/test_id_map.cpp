#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "lob/id_map.hpp"

namespace {

using lob::IdMap;
using lob::OrderId;

TEST(IdMap, InsertFindErase) {
  IdMap<int> m(4);
  int a = 1, b = 2;
  EXPECT_TRUE(m.insert(10, &a));
  EXPECT_TRUE(m.insert(0, &b));  // id 0 is an ordinary key
  EXPECT_EQ(m.find(10), &a);
  EXPECT_EQ(m.find(0), &b);
  EXPECT_EQ(m.find(11), nullptr);
  EXPECT_EQ(m.size(), 2u);
  EXPECT_TRUE(m.erase(10));
  EXPECT_FALSE(m.erase(10));
  EXPECT_EQ(m.find(10), nullptr);
  EXPECT_EQ(m.find(0), &b);
  EXPECT_EQ(m.size(), 1u);
}

TEST(IdMap, DuplicateInsertChangesNothing) {
  IdMap<int> m;
  int a = 1, b = 2;
  EXPECT_TRUE(m.insert(5, &a));
  EXPECT_FALSE(m.insert(5, &b));
  EXPECT_EQ(m.find(5), &a);
  EXPECT_EQ(m.size(), 1u);
}

TEST(IdMap, ReserveStopsRehashing) {
  IdMap<int> m(1);
  m.reserve(1000);
  const auto table = m.table_size();
  int v = 0;
  for (OrderId k = 0; k < 1000; ++k) m.insert(k, &v);
  EXPECT_EQ(m.table_size(), table);
  EXPECT_LE(2 * m.size(), m.table_size());  // load factor stays <= 1/2
}

// Differential test against std::unordered_map. A small key range forces long probe chains
// and constant erase/reinsert churn, which is what exercises backward-shift deletion; the
// tiny starting table forces several rehashes on the way.
TEST(IdMap, RandomOperationsMatchStdUnorderedMap) {
  std::mt19937_64 rng(20261119);
  std::uniform_int_distribution<OrderId> key(0, 300);
  std::uniform_int_distribution<int> op(0, 2);
  std::vector<int> values(301);

  IdMap<int> m(2);
  std::unordered_map<OrderId, int*> ref;
  for (int step = 0; step < 200'000; ++step) {
    const OrderId k = key(rng);
    switch (op(rng)) {
      case 0: {
        const bool fresh = ref.find(k) == ref.end();
        ASSERT_EQ(m.insert(k, &values[k]), fresh) << "step " << step;
        ref.emplace(k, &values[k]);
        break;
      }
      case 1:
        ASSERT_EQ(m.erase(k), ref.erase(k) == 1) << "step " << step;
        break;
      default: {
        auto it = ref.find(k);
        ASSERT_EQ(m.find(k), it == ref.end() ? nullptr : it->second) << "step " << step;
      }
    }
    ASSERT_EQ(m.size(), ref.size());
  }
  // Final sweep: every key agrees.
  for (OrderId k = 0; k <= 300; ++k) {
    auto it = ref.find(k);
    EXPECT_EQ(m.find(k), it == ref.end() ? nullptr : it->second);
  }
}

}  // namespace
