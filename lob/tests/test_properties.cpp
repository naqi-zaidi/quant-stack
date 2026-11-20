// Property-based tests (Phase 2, step 2). Seeded random flows of every order type, cancels,
// modifies and malformed input, across five market regimes:
//   - the engine satisfies every property after every call, and its event stream agrees
//     with its book;
//   - the engine matches a naive reference model call for call;
//   - twelve planted bugs are each caught, and shrunk to a counterexample of a few calls.
// lob_property_report runs the same checks at larger scale and prints the README tables.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "property_harness.hpp"

namespace lob::testing_helpers {
namespace {

// Kept small so the suite stays quick under ASan (about 30x slower); lob_property_report
// runs the same checks on 400,000 calls in the Release CI job.
constexpr std::uint64_t kSeedsPerRegime = 3;
constexpr std::size_t kCallsPerFlow = 1000;
constexpr std::size_t kHuntCalls = 2000;

// On failure: the shrunk counterexample, replayed call by call on the engine.
std::string engine_counterexample(std::vector<Op> ops, const FlowResult& fr) {
  ops.resize(fr.step + 1);
  const std::string prop = fr.violation->property;
  ops = shrink(ops, [&](const std::vector<Op>& o) {
    EngineUnderTest e;
    const FlowResult r = run_properties(o, e);
    return r.violation && r.violation->property == prop;
  });
  EngineUnderTest e;
  return "minimal counterexample:\n" + describe(ops, e);
}

TEST(Properties, EngineHoldsEveryPropertyOnEveryCall) {
  for (const Regime& rg : kRegimes) {
    for (std::uint64_t seed = 1; seed <= kSeedsPerRegime; ++seed) {
      const std::vector<Op> ops = generate_flow(rg, seed, kCallsPerFlow);
      EngineUnderTest eng;
      const FlowResult fr = run_properties(ops, eng);
      ASSERT_FALSE(fr.violation) << rg.name << ", seed " << seed << ", call " << fr.step + 1
                                 << ": " << fr.violation->property << ": "
                                 << fr.violation->detail << "\n"
                                 << engine_counterexample(ops, fr);
      EXPECT_EQ(fr.calls, kCallsPerFlow);
      EXPECT_GT(fr.trades, kCallsPerFlow / 50) << rg.name << ": flow should trade";
    }
  }
}

TEST(Properties, EngineMatchesTheReferenceModelCallForCall) {
  for (const Regime& rg : kRegimes) {
    for (std::uint64_t seed = 1; seed <= kSeedsPerRegime; ++seed) {
      const std::vector<Op> ops = generate_flow(rg, seed, kCallsPerFlow);
      const FlowResult fr = run_differential(ops);
      ASSERT_FALSE(fr.violation) << rg.name << ", seed " << seed << ", call " << fr.step + 1
                                 << ": " << fr.violation->detail;
    }
  }
}

// The oracle must itself satisfy the properties, or the differential test proves nothing.
TEST(Properties, ReferenceModelHoldsEveryProperty) {
  for (const Regime& rg : kRegimes) {
    const std::vector<Op> ops = generate_flow(rg, 99, kCallsPerFlow);
    const FlowResult fr = run_properties(ops, Mutation::None);
    ASSERT_FALSE(fr.violation) << rg.name << ": " << fr.violation->property << ": "
                               << fr.violation->detail;
  }
}

TEST(Properties, FlowsAreReproducibleFromTheirSeed) {
  auto fingerprint = [](const std::vector<Op>& ops) {
    std::vector<std::int64_t> f;
    for (const Op& op : ops) {
      f.push_back(static_cast<std::int64_t>(op.kind));
      f.push_back(op.order.price.ticks);
      f.push_back(op.order.qty);
      f.push_back(static_cast<std::int64_t>(op.target));
      f.push_back(op.delta);
    }
    return f;
  };
  const Regime& rg = kRegimes[0];
  EXPECT_EQ(fingerprint(generate_flow(rg, 7, 500)), fingerprint(generate_flow(rg, 7, 500)));
  EXPECT_NE(fingerprint(generate_flow(rg, 7, 500)), fingerprint(generate_flow(rg, 8, 500)));
}

TEST(Properties, FlowsExerciseEveryKindOfCall) {
  std::size_t kinds[3] = {0, 0, 0};
  std::size_t types[5] = {0, 0, 0, 0, 0};
  std::size_t bad = 0;
  for (const Regime& rg : kRegimes) {
    for (const Op& op : generate_flow(rg, 1, kCallsPerFlow)) {
      ++kinds[static_cast<int>(op.kind)];
      if (op.kind == Op::Kind::Submit) {
        ++types[static_cast<int>(op.order.type)];
        bad += (op.order.qty <= 0 || op.order.price.ticks <= 0) ? 1 : 0;
      }
      bad += (op.kind == Op::Kind::Modify && op.mod == Op::Mod::Bad) ? 1 : 0;
    }
  }
  for (std::size_t k : kinds) EXPECT_GT(k, 500u);
  for (std::size_t t : types) EXPECT_GT(t, 100u);
  EXPECT_GT(bad, 30u);
}

// ---------------------------------------------------------------- planted bugs

class SeededBugTest : public ::testing::TestWithParam<SeededBug> {};

TEST_P(SeededBugTest, IsCaughtAndShrunkToAFewCalls) {
  const SeededBug& bug = GetParam();
  const Hunt h = hunt(bug.m, 10, kHuntCalls);
  ASSERT_TRUE(h.caught) << bug.name << " survived " << h.flows << " flows";
  EXPECT_GT(h.differential_at, 0u) << "the reference comparison missed " << bug.name;

  // Shrunk to a handful of calls, and minimal: dropping any single call hides the bug.
  EXPECT_LE(h.shrunk.size(), 4u) << describe(h.shrunk, bug.m);
  EXPECT_TRUE(breaks(h.shrunk, bug.m, h.violation.property));
  for (std::size_t i = 0; i < h.shrunk.size(); ++i) {
    std::vector<Op> fewer = h.shrunk;
    fewer.erase(fewer.begin() + static_cast<std::ptrdiff_t>(i));
    EXPECT_FALSE(breaks(fewer, bug.m, h.violation.property)) << "call " << i + 1 << " is not needed";
  }
  // The correct model passes the very flow that exposes the bug.
  EXPECT_FALSE(run_properties(h.shrunk, Mutation::None).violation);
}

INSTANTIATE_TEST_SUITE_P(AllBugs, SeededBugTest, ::testing::ValuesIn(kSeededBugs),
                         [](const ::testing::TestParamInfo<SeededBug>& p) {
                           return std::string(p.param.key);
                         });

}  // namespace
}  // namespace lob::testing_helpers
