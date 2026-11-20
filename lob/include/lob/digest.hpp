#pragma once

// FNV-1a 64-bit digest of an event stream, for pinning behaviour in tests.
//
// Each event is serialized canonically: a one-byte tag, then every field as an 8-byte
// little-endian integer, in declaration order. The bytes depend only on the event values,
// never on struct layout, endianness or compiler, so a digest computed in the sandbox matches
// one computed in CI. Order matters: the digest covers the stream in arrival order.
//
// Convention (see README): a pinned digest that changes is a behaviour change. The commit
// that changes it must say why; an unexplained change is a bug.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "lob/events.hpp"

namespace lob {

class Fnv1a64 {
 public:
  static constexpr std::uint64_t kOffset = 14695981039346656037ULL;
  static constexpr std::uint64_t kPrime = 1099511628211ULL;

  constexpr void byte(std::uint8_t b) noexcept {
    h_ ^= b;
    h_ *= kPrime;
  }
  constexpr void bytes(std::string_view s) noexcept {
    for (char c : s) byte(static_cast<std::uint8_t>(c));
  }
  constexpr void u64(std::uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i) byte(static_cast<std::uint8_t>(v >> (8 * i)));
  }
  constexpr void i64(std::int64_t v) noexcept { u64(static_cast<std::uint64_t>(v)); }
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return h_; }

 private:
  std::uint64_t h_{kOffset};
};

// Listener that folds every event into a running FNV-1a digest.
class EventDigest final : public Listener {
 public:
  void on_trade(const Trade& t) override {
    h_.byte('T');
    h_.u64(t.taker);
    h_.u64(t.maker);
    h_.u64(static_cast<std::uint64_t>(t.taker_side));
    h_.i64(t.price.ticks);
    h_.i64(t.qty);
    h_.u64(t.ts);
    ++events_;
  }
  void on_level_update(const LevelUpdate& u) override {
    h_.byte('L');
    h_.u64(static_cast<std::uint64_t>(u.side));
    h_.i64(u.price.ticks);
    h_.i64(u.qty);
    h_.u64(u.count);
    h_.u64(u.ts);
    ++events_;
  }
  void on_report(const ExecReport& r) override {
    h_.byte('R');
    h_.u64(r.id);
    h_.u64(static_cast<std::uint64_t>(r.type));
    h_.u64(static_cast<std::uint64_t>(r.reason));
    h_.u64(static_cast<std::uint64_t>(r.side));
    h_.i64(r.price.ticks);
    h_.i64(r.last_qty);
    h_.i64(r.leaves);
    h_.u64(r.ts);
    ++events_;
  }

  [[nodiscard]] std::uint64_t value() const noexcept { return h_.value(); }
  [[nodiscard]] std::size_t events() const noexcept { return events_; }

 private:
  Fnv1a64 h_;
  std::size_t events_{0};
};

}  // namespace lob
