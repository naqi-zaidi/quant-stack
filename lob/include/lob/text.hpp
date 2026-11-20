#pragma once

// Deterministic text forms of the book and its event stream, for snapshot ("golden") tests
// and debugging. The same book state or event always prints the same bytes, on any platform.
//
//   to_text(book)   a ladder: asks worst to best, a rule, bids best to worst; every level
//                   shows its total, order count and FIFO queue as id:qty
//   to_text(event)  one line per Trade, LevelUpdate or ExecReport
//   TextLog         a Listener that records events as lines, in arrival order

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "lob/events.hpp"
#include "lob/types.hpp"

namespace lob {

[[nodiscard]] inline std::string to_text(const Trade& t) {
  std::ostringstream os;
  os << "trade  taker=" << t.taker << " maker=" << t.maker << " " << to_string(t.taker_side)
     << " " << t.qty << " @ " << t.price.ticks << " ts=" << t.ts;
  return os.str();
}

[[nodiscard]] inline std::string to_text(const LevelUpdate& u) {
  std::ostringstream os;
  os << "level  " << to_string(u.side) << " " << u.price.ticks << " qty=" << u.qty
     << " orders=" << u.count << " ts=" << u.ts;
  return os.str();
}

[[nodiscard]] inline std::string to_text(const ExecReport& r) {
  std::ostringstream os;
  os << "report id=" << r.id << " " << to_string(r.type);
  if (r.reason != Reason::None) os << "(" << to_string(r.reason) << ")";
  os << " " << to_string(r.side) << " px=" << r.price.ticks << " last=" << r.last_qty
     << " leaves=" << r.leaves << " ts=" << r.ts;
  return os.str();
}

// Works with any book exposing orders(Side): every resting order in priority order.
template <typename Book>
[[nodiscard]] std::string to_text(const Book& book) {
  std::ostringstream os;
  auto ladder = [&](Side s) {
    const auto orders = book.orders(s);
    // Group consecutive orders by price: orders() is already best price first, FIFO within.
    std::vector<std::string> rows;
    for (std::size_t i = 0; i < orders.size();) {
      std::size_t j = i;
      Quantity total = 0;
      std::ostringstream q;
      for (; j < orders.size() && orders[j].price == orders[i].price; ++j) {
        total += orders[j].qty;
        q << (j == i ? "" : " ") << orders[j].id << ":" << orders[j].qty;
      }
      std::ostringstream row;
      row << (s == Side::Sell ? "ask " : "bid ") << orders[i].price.ticks << "  " << total
          << " in " << (j - i) << "  [" << q.str() << "]";
      rows.push_back(row.str());
      i = j;
    }
    return rows;
  };
  const auto asks = ladder(Side::Sell);
  for (auto it = asks.rbegin(); it != asks.rend(); ++it) os << "  " << *it << "\n";
  if (asks.empty()) os << "  (no asks)\n";
  os << "  ----\n";
  for (const auto& r : ladder(Side::Buy)) os << "  " << r << "\n";
  if (book.orders(Side::Buy).empty()) os << "  (no bids)\n";
  return os.str();
}

// Records every event as a line of text, in the order the book emitted them.
class TextLog final : public Listener {
 public:
  void on_trade(const Trade& t) override { lines.push_back(to_text(t)); }
  void on_level_update(const LevelUpdate& u) override { lines.push_back(to_text(u)); }
  void on_report(const ExecReport& r) override { lines.push_back(to_text(r)); }

  // Lines recorded since the last call, then forgets them.
  std::string take(const char* indent = "  ") {
    std::string out;
    for (const auto& l : lines) out += indent + l + "\n";
    lines.clear();
    return out;
  }

  std::vector<std::string> lines;
};

}  // namespace lob
