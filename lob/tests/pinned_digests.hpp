#pragma once

// Pinned FNV-1a 64 digests of the book's event stream (see lob/digest.hpp).
//
// RULE: if a pinned digest changes, the commit that changes it must say why. An unexplained
// change is a bug. A test failure prints the new digest and event count to paste here.
//
// Generated at Phase 2 step 3.5 on the std::map book (parent commit 8be5b97). Every later
// level structure must reproduce them byte for byte.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <ios>
#include <string_view>

#include "lob/digest.hpp"

namespace lob::testing_helpers {

struct PinnedDigest {
  std::string_view name;
  std::uint64_t digest;
  std::size_t events;
};

// flow/*: seed 1, 2,000 calls per regime (test_digest.cpp). lobster/*: the hand-built
// replays in test_lobster.cpp, seeding and level rebuilds included.
inline constexpr std::array<PinnedDigest, 18> kPinnedDigests{{
    {"flow/tight spread", 0xbc5e276a2c6c864fULL, 4994},
    {"flow/deep queues", 0xeaf1f47c7e8acbf6ULL, 4554},
    {"flow/wide book", 0x36463a0a656ca4bbULL, 4718},
    {"flow/cancel churn", 0x8795ef0fadb730d4ULL, 3807},
    {"flow/sweeps", 0x4fdb74bd6ed8b05eULL, 4491},
    {"lobster/consistent_day", 0xf4560d597b3dde88ULL, 13},
    {"lobster/corrupted_snapshot", 0xf4560d597b3dde88ULL, 13},  // same stream: only the snapshot differs
    {"lobster/missing_message", 0xe4707692489ccb06ULL, 9},
    {"lobster/pre_open_orders", 0xad30e0e83b7ed7d2ULL, 11},
    {"lobster/out_of_order_execution", 0x967caf4a7701fc2bULL, 6},
    {"lobster/level_scrolls_into_view", 0xca010c6f5fcdaa7bULL, 15},
    {"lobster/stale_level_returns", 0xfa0a7c141393a43aULL, 10},
    {"lobster/already_visible_level", 0xacc26f690159eda0ULL, 4},
    {"lobster/error_crossing_submission", 0x8bf372a8108507bdULL, 3},
    {"lobster/error_unknown_order", 0x96b36014588ea55aULL, 2},
    {"lobster/error_delete_size", 0xfe2d80b222207c8dULL, 2},
    {"lobster/error_oversize_execution", 0xfe2d80b222207c8dULL, 2},  // same: the bad event applies nothing
    {"lobster/files_day", 0xbcc6f3205fa6d2c5ULL, 13},
}};

inline void expect_pinned(std::string_view name, const EventDigest& d) {
  for (const PinnedDigest& p : kPinnedDigests) {
    if (p.name != name) continue;
    EXPECT_EQ(d.value(), p.digest)
        << name << ": event-stream digest changed to 0x" << std::hex << d.value() << std::dec
        << " (" << d.events() << " events). If intended, update pinned_digests.hpp and say why "
        << "in the commit message.";
    EXPECT_EQ(d.events(), p.events) << name;
    return;
  }
  ADD_FAILURE() << "no pinned digest for " << name << "; add:\n    {\"" << name << "\", 0x"
                << std::hex << d.value() << std::dec << "ULL, " << d.events() << "},";
}

}  // namespace lob::testing_helpers
