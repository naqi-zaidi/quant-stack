#pragma once

// What the order book tells the outside world. Three streams, all stamped with the event
// time of the message that caused them:
//
//   Trade        one per maker order matched (price = the resting order's price)
//   LevelUpdate  new aggregate size of a price level after it changed (qty 0 = level gone),
//                enough for a subscriber to maintain an L2 book incrementally
//   ExecReport   per-order lifecycle: New, Fill, Cancelled, Replaced, Rejected

#include <cstdint>
#include <string_view>
#include <vector>

#include "lob/types.hpp"

namespace lob {

enum class ExecType : std::uint8_t { New, Fill, Cancelled, Replaced, Rejected };

enum class Reason : std::uint8_t {
  None,
  DuplicateOrderId,       // an order with this id is already live
  InvalidQuantity,        // quantity <= 0
  InvalidPrice,           // price <= 0 on an order that needs a limit
  UnknownOrder,           // cancel/modify of an id that is not resting
  WouldCross,             // post-only order (or re-price) would take liquidity
  InsufficientLiquidity,  // fill-or-kill could not be filled in full
  UserCancel,             // explicit cancel
  Unfilled,               // market/IOC remainder expired instead of resting
};

struct Trade {
  OrderId taker{};
  OrderId maker{};
  Side taker_side{Side::Buy};
  Price price{};
  Quantity qty{};
  Timestamp ts{};

  friend bool operator==(const Trade&, const Trade&) = default;
};

struct LevelUpdate {
  Side side{Side::Buy};
  Price price{};
  Quantity qty{};             // aggregate resting size after the change; 0 = level removed
  std::uint32_t count{0};     // resting orders at the level after the change
  Timestamp ts{};

  friend bool operator==(const LevelUpdate&, const LevelUpdate&) = default;
};

struct ExecReport {
  OrderId id{};
  ExecType type{ExecType::New};
  Reason reason{Reason::None};
  Side side{Side::Buy};
  Price price{};        // limit price (New/Replaced/Cancelled), trade price (Fill)
  Quantity last_qty{};  // traded (Fill) or cancelled (Cancelled) by this event
  Quantity leaves{};    // still open after this event
  Timestamp ts{};

  friend bool operator==(const ExecReport&, const ExecReport&) = default;
};

// Subscriber interface. Defaults are no-ops so a listener overrides only what it needs.
// Callbacks run synchronously inside the book call; they must not call back into the book.
class Listener {
 public:
  virtual ~Listener() = default;
  virtual void on_trade(const Trade&) {}
  virtual void on_level_update(const LevelUpdate&) {}
  virtual void on_report(const ExecReport&) {}
};

// Records every event in arrival order. Used by the tests and, later, the Python bindings.
class EventLog final : public Listener {
 public:
  void on_trade(const Trade& t) override { trades.push_back(t); }
  void on_level_update(const LevelUpdate& u) override { updates.push_back(u); }
  void on_report(const ExecReport& r) override { reports.push_back(r); }

  void clear() noexcept {
    trades.clear();
    updates.clear();
    reports.clear();
  }

  std::vector<Trade> trades;
  std::vector<LevelUpdate> updates;
  std::vector<ExecReport> reports;
};

[[nodiscard]] constexpr std::string_view to_string(Side s) noexcept {
  return s == Side::Buy ? "Buy" : "Sell";
}

[[nodiscard]] constexpr std::string_view to_string(OrderType t) noexcept {
  switch (t) {
    case OrderType::Limit: return "Limit";
    case OrderType::Market: return "Market";
    case OrderType::IOC: return "IOC";
    case OrderType::FOK: return "FOK";
    case OrderType::PostOnly: return "PostOnly";
  }
  return "?";
}

[[nodiscard]] constexpr std::string_view to_string(ExecType t) noexcept {
  switch (t) {
    case ExecType::New: return "New";
    case ExecType::Fill: return "Fill";
    case ExecType::Cancelled: return "Cancelled";
    case ExecType::Replaced: return "Replaced";
    case ExecType::Rejected: return "Rejected";
  }
  return "?";
}

[[nodiscard]] constexpr std::string_view to_string(Reason r) noexcept {
  switch (r) {
    case Reason::None: return "None";
    case Reason::DuplicateOrderId: return "DuplicateOrderId";
    case Reason::InvalidQuantity: return "InvalidQuantity";
    case Reason::InvalidPrice: return "InvalidPrice";
    case Reason::UnknownOrder: return "UnknownOrder";
    case Reason::WouldCross: return "WouldCross";
    case Reason::InsufficientLiquidity: return "InsufficientLiquidity";
    case Reason::UserCancel: return "UserCancel";
    case Reason::Unfilled: return "Unfilled";
  }
  return "?";
}

}  // namespace lob
