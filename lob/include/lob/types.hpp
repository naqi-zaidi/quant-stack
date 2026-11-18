#pragma once

// Core value types shared by the order book, the feed replay and the Python bindings.
// Prices are integer ticks, never floating point: equality and ordering must be exact
// for price-time priority and for reconciling against exchange data.

#include <compare>
#include <cstdint>

namespace lob {

using OrderId = std::uint64_t;
using Quantity = std::int64_t;
using Timestamp = std::uint64_t;  // nanoseconds after midnight, exchange event time

// LOBSTER quotes prices as dollars * 10'000, so one tick here is $0.0001.
inline constexpr std::int64_t kLobsterPriceScale = 10'000;

struct Price {
  std::int64_t ticks{0};

  [[nodiscard]] static constexpr Price from_lobster(std::int64_t raw) noexcept {
    return Price{raw};
  }
  [[nodiscard]] constexpr double to_dollars() const noexcept {
    return static_cast<double>(ticks) / static_cast<double>(kLobsterPriceScale);
  }

  constexpr auto operator<=>(const Price&) const = default;
};

enum class Side : std::uint8_t { Buy, Sell };

[[nodiscard]] constexpr Side opposite(Side side) noexcept {
  return side == Side::Buy ? Side::Sell : Side::Buy;
}

// True when an incoming order on `side` with limit `limit` can trade against a
// resting order at `resting`. A buy crosses at or above the ask; a sell at or below the bid.
[[nodiscard]] constexpr bool crosses(Side side, Price limit, Price resting) noexcept {
  return side == Side::Buy ? limit >= resting : limit <= resting;
}

// Ranks resting prices for priority: the better price for that side compares first.
// Bids: higher is better. Asks: lower is better.
[[nodiscard]] constexpr bool better(Side side, Price a, Price b) noexcept {
  return side == Side::Buy ? a > b : a < b;
}

enum class OrderType : std::uint8_t {
  Limit,     // rests if not fully filled
  Market,    // takes liquidity at any price, never rests
  IOC,       // immediate-or-cancel: fills what it can, cancels the rest
  FOK,       // fill-or-kill: fills entirely or not at all
  PostOnly,  // rejected if it would cross, so it only ever adds liquidity
};

struct Order {
  OrderId id{};
  Side side{Side::Buy};
  OrderType type{OrderType::Limit};
  Price price{};
  Quantity qty{};
  Timestamp ts{};
};

}  // namespace lob
