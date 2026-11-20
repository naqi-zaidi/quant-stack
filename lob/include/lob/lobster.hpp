#pragma once

// LOBSTER replay: rebuild Nasdaq's book from a LOBSTER message file with the matching
// engine, and compare it with LOBSTER's own book file after every event.
//
// A LOBSTER day is two CSV files with one row per event:
//   message    time, type, order id, size, price, direction
//   orderbook  the book after that event, `levels` deep: ask px, ask size, bid px, bid size, ...
// Prices are dollars x 10,000, the same units as lob::Price, so nothing is converted.
//
// What the replay has to deal with, and how:
//   - The day starts with orders already resting (submitted before 09:30). The first
//     snapshot seeds each visible level as one block of anonymous liquidity.
//   - A level-N file only carries messages for orders inside the top N levels. Orders
//     outside the window come and go unseen, so when a price first scrolls into view (it was
//     beyond the deepest visible level on the previous event), that level is rebuilt as one
//     anonymous block of the size the snapshot shows. Only newly visible prices are ever
//     touched; every level that was already visible must match from messages alone.
//   - Later events naming an order the replay never saw (one that rested before the open or
//     arrived out of view) draw on the anonymous block at that price. Anonymous blocks are
//     always older than any order that joins the level after them, so FIFO is preserved.
//   - Submissions go in as post-only: a submission that would cross shows the book is wrong,
//     and the engine refuses it instead of trading.
//   - LOBSTER records an execution as a reduction of the resting order it hit. When that
//     order is first in the queue at the best price, the replay sends an aggressive IOC order
//     through the matching engine and requires it to fill exactly that order. When it is
//     not (Nasdaq filled a later order first), the named order is reduced directly and the
//     event is counted, since book state must follow what actually happened.
//   - Hidden executions (type 5) do not change the visible book and are skipped.

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "lob/events.hpp"
#include "lob/order_book.hpp"
#include "lob/types.hpp"

namespace lob::lobster {

enum class Event : std::uint8_t {
  Submit = 1,          // new limit order
  Cancel = 2,          // partial cancellation
  Delete = 3,          // full deletion
  ExecuteVisible = 4,  // execution against a visible resting order
  ExecuteHidden = 5,   // execution against a hidden order: no visible change
  Cross = 6,           // auction cross
  Halt = 7,            // trading halt indicator
};

[[nodiscard]] std::string_view to_string(Event e) noexcept;

struct Message {
  Timestamp ns{};  // nanoseconds after midnight
  Event event{Event::Submit};
  OrderId id{};
  Quantity size{};
  Price price{};
  Side side{Side::Buy};  // side of the resting order (direction 1 = buy, -1 = sell)
};

struct DepthLevel {
  Price price{};
  Quantity qty{};
  friend bool operator==(const DepthLevel&, const DepthLevel&) = default;
};

// One orderbook row: occupied levels only, best first.
struct Snapshot {
  std::vector<DepthLevel> asks;
  std::vector<DepthLevel> bids;
  [[nodiscard]] const std::vector<DepthLevel>& side(Side s) const {
    return s == Side::Buy ? bids : asks;
  }
};

// Parse one line of each file. Throw std::runtime_error on malformed input.
[[nodiscard]] Message parse_message(std::string_view line);
void parse_snapshot(std::string_view line, std::size_t levels, Snapshot& out);

// Facts encoded in a LOBSTER file name such as
// AAPL_2012-06-21_34200000_57600000_message_10.csv.
struct FileInfo {
  std::string ticker;
  std::string date;
  std::size_t levels{0};
  std::string orderbook_path;  // the matching orderbook file in the same folder
};
[[nodiscard]] FileInfo describe_file(const std::string& message_path);

// Reads a message file and its orderbook file in lockstep, one event at a time.
class DayReader {
 public:
  DayReader(const std::string& message_path, const std::string& orderbook_path,
            std::size_t levels);
  // Next event and the book after it. False at the end of both files; throws if one file
  // ends before the other or a line is malformed.
  bool next(Message& m, Snapshot& s);
  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

 private:
  std::ifstream msg_;
  std::ifstream book_;
  std::string line_;
  std::size_t levels_;
  std::size_t rows_{0};
};

struct ReplayStats {
  std::size_t events{0};                // events replayed (after the seeding snapshot)
  std::array<std::size_t, 8> by_type{}; // indexed by Event value
  std::size_t mismatches{0};            // events after which any visible level differed
  std::size_t first_mismatch{0};        // 1-based event number, 0 = none
  std::string first_mismatch_detail;
  std::size_t errors{0};                // events that could not be applied consistently
  std::size_t first_error{0};
  std::string first_error_detail;
  std::size_t seeded_levels{0};
  std::size_t revealed_levels{0};       // levels rebuilt as they scrolled into view
  std::size_t revealed_already_right{0};  // ... where the replayed size was already right
  std::size_t unknown_id_events{0};     // events drawn from anonymous liquidity
  std::size_t executions_matched{0};    // type 4 replayed through the matcher, exact fill
  std::size_t executions_out_of_order{0};  // named order was not first in the queue
  std::size_t comparisons{0};           // level comparisons made (price and size each)
};

class Replay {
 public:
  explicit Replay(std::size_t levels);

  // Seeds the book from the snapshot that follows the first message of the day.
  void seed(const Snapshot& s, Timestamp ts = 0);

  // Applies one event, syncs newly visible levels, then compares every visible level with
  // `after`. Returns true if they all match and the event applied cleanly.
  bool step(const Message& m, const Snapshot& after);

  [[nodiscard]] const ReplayStats& stats() const noexcept { return st_; }
  [[nodiscard]] const OrderBook& book() const noexcept { return book_; }

 private:
  class TradeTap final : public Listener {
   public:
    void on_trade(const Trade& t) override { trades.push_back(t); }
    std::vector<Trade> trades;
  };

  static constexpr std::size_t idx(Side s) noexcept { return s == Side::Buy ? 0 : 1; }

  [[nodiscard]] std::optional<OrderId> anon_at(Side s, Price p);
  void add_anon(Side s, Price p, Quantity q, Timestamp ts);
  void clear_level(Side s, Price p, Timestamp ts);
  void reduce(OrderId id, Quantity by, Timestamp ts);
  void execute(const Message& m, OrderId target);
  void sync_revealed(const Snapshot& after, Timestamp ts);
  bool compare(const Snapshot& after);
  void fail(std::string what);

  std::size_t levels_;
  TradeTap tap_;
  OrderBook book_;
  std::map<Price, OrderId> anon_[2];
  std::optional<Price> boundary_[2];  // deepest visible price; nullopt = whole side visible
  OrderId next_anon_;
  OrderId next_taker_;
  ReplayStats st_;
  bool clean_{true};  // current event applied without error
};

}  // namespace lob::lobster
