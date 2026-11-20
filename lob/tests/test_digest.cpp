// Pinned event-stream digests for the fixed-seed property flows (one per regime). The hand-
// built LOBSTER replays pin theirs in test_lobster.cpp; all pins live in pinned_digests.hpp.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "lob/digest.hpp"
#include "lob/order_book.hpp"
#include "pinned_digests.hpp"
#include "property_harness.hpp"

namespace lob::testing_helpers {
namespace {

constexpr std::size_t kDigestCalls = 2000;

TEST(Fnv1a64, MatchesPublishedTestVectors) {
  auto of = [](std::string_view s) {
    Fnv1a64 h;
    h.bytes(s);
    return h.value();
  };
  EXPECT_EQ(of(""), 0xcbf29ce484222325ULL);
  EXPECT_EQ(of("a"), 0xaf63dc4c8601ec8cULL);
  EXPECT_EQ(of("foobar"), 0x85944171f73967e8ULL);
}

TEST(EventDigest, DependsOnEveryFieldAndOnOrder) {
  const Trade t{1, 2, Side::Buy, Price{100}, 5, 7};
  const LevelUpdate u{Side::Sell, Price{100}, 0, 0, 7};
  EventDigest a, b, c, d;
  a.on_trade(t);
  a.on_level_update(u);
  b.on_level_update(u);
  b.on_trade(t);
  Trade t2 = t;
  t2.qty = 6;
  c.on_trade(t2);
  c.on_level_update(u);
  d.on_trade(t);
  d.on_level_update(u);
  EXPECT_NE(a.value(), b.value());  // order matters
  EXPECT_NE(a.value(), c.value());  // one field differs
  EXPECT_EQ(a.value(), d.value());  // same stream, same digest
  EXPECT_EQ(a.events(), 2u);
}

// Drives a book with a digest listener through a seeded flow, resolving each op against the
// book's own state, exactly as the property tests do.
template <typename Book>
EventDigest digest_flow(const Regime& rg, std::uint64_t seed) {
  EventDigest dg;
  Book book(&dg, 64);
  const std::vector<Op> ops = generate_flow(rg, seed, kDigestCalls);
  for (std::size_t i = 0; i < ops.size(); ++i) {
    const BookState st{book.orders(Side::Buy), book.orders(Side::Sell)};
    const Call c = resolve(ops[i], st, static_cast<Timestamp>(i));
    switch (c.kind) {
      case Op::Kind::Submit: book.submit(c.order); break;
      case Op::Kind::Cancel: book.cancel(c.order.id, c.order.ts); break;
      case Op::Kind::Modify: book.modify(c.order.id, c.order.price, c.order.qty, c.order.ts); break;
    }
  }
  return dg;
}

TEST(PinnedDigests, PropertyFlowsInEveryRegime) {
  for (const Regime& rg : kRegimes) {
    expect_pinned("flow/" + std::string(rg.name), digest_flow<OrderBook>(rg, 1));
  }
}

}  // namespace
}  // namespace lob::testing_helpers
