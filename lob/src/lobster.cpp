#include "lob/lobster.hpp"

#include <charconv>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace lob::lobster {

namespace {

constexpr std::int64_t kEmptyAsk = 9'999'999'999;
constexpr std::int64_t kEmptyBid = -9'999'999'999;
// Ids the replay invents. LOBSTER order ids are assigned by Nasdaq in the order flow and stay
// far below 2^62, so these ranges can never collide with a real id.
constexpr OrderId kAnonBase = OrderId{1} << 62;
constexpr OrderId kTakerBase = OrderId{3} << 62;

[[noreturn]] void bad(std::string_view what, std::string_view line) {
  throw std::runtime_error(std::string(what) + ": '" + std::string(line.substr(0, 120)) + "'");
}

std::string_view trim_cr(std::string_view s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.remove_suffix(1);
  return s;
}

// Splits on commas into `out`; returns the number of fields.
std::size_t split(std::string_view line, std::string_view* out, std::size_t max) {
  std::size_t n = 0;
  while (n < max) {
    const std::size_t comma = line.find(',');
    out[n++] = line.substr(0, comma);
    if (comma == std::string_view::npos) return n;
    line.remove_prefix(comma + 1);
  }
  return n;
}

std::int64_t to_int(std::string_view field, std::string_view line) {
  std::int64_t v = 0;
  const auto [p, ec] = std::from_chars(field.data(), field.data() + field.size(), v);
  if (ec != std::errc{} || p != field.data() + field.size()) bad("not an integer", line);
  return v;
}

// "34200.017459617" -> 34200017459617 ns. Exact: no floating point.
Timestamp to_ns(std::string_view field, std::string_view line) {
  const std::size_t dot = field.find('.');
  const std::int64_t secs = to_int(field.substr(0, dot), line);
  std::int64_t frac = 0;
  int digits = 0;
  if (dot != std::string_view::npos) {
    for (char c : field.substr(dot + 1)) {
      if (c < '0' || c > '9') bad("bad timestamp", line);
      if (digits < 9) {
        frac = frac * 10 + (c - '0');
        ++digits;
      }
    }
  }
  for (; digits < 9; ++digits) frac *= 10;
  if (secs < 0) bad("negative timestamp", line);
  return static_cast<Timestamp>(secs) * 1'000'000'000ULL + static_cast<Timestamp>(frac);
}

}  // namespace

std::string_view to_string(Event e) noexcept {
  switch (e) {
    case Event::Submit: return "submit";
    case Event::Cancel: return "partial cancel";
    case Event::Delete: return "delete";
    case Event::ExecuteVisible: return "visible execution";
    case Event::ExecuteHidden: return "hidden execution";
    case Event::Cross: return "cross";
    case Event::Halt: return "halt";
  }
  return "?";
}

// ---------------------------------------------------------------- parsing

Message parse_message(std::string_view raw) {
  const std::string_view line = trim_cr(raw);
  std::array<std::string_view, 8> f{};
  if (split(line, f.data(), f.size()) < 6) bad("message line needs 6 fields", line);
  Message m;
  m.ns = to_ns(f[0], line);
  const std::int64_t type = to_int(f[1], line);
  if (type < 1 || type > 7) bad("unknown event type", line);
  m.event = static_cast<Event>(type);
  const std::int64_t id = to_int(f[2], line);
  if (id < 0) bad("negative order id", line);
  m.id = static_cast<OrderId>(id);
  m.size = to_int(f[3], line);
  m.price = Price{to_int(f[4], line)};
  const std::int64_t dir = to_int(f[5], line);
  if (dir != 1 && dir != -1) bad("direction must be 1 or -1", line);
  m.side = dir == 1 ? Side::Buy : Side::Sell;
  return m;
}

void parse_snapshot(std::string_view raw, std::size_t levels, Snapshot& out) {
  const std::string_view line = trim_cr(raw);
  out.asks.clear();
  out.bids.clear();
  std::string_view rest = line;
  bool asks_done = false, bids_done = false;
  for (std::size_t i = 0; i < levels; ++i) {
    std::array<std::string_view, 4> f{};
    for (std::size_t j = 0; j < 4; ++j) {
      if (rest.data() == nullptr) bad("orderbook line has too few fields", line);
      const std::size_t comma = rest.find(',');
      f[j] = rest.substr(0, comma);
      rest = comma == std::string_view::npos ? std::string_view{} : rest.substr(comma + 1);
    }
    const std::int64_t ap = to_int(f[0], line), as = to_int(f[1], line);
    const std::int64_t bp = to_int(f[2], line), bs = to_int(f[3], line);
    if (ap == kEmptyAsk) {
      asks_done = true;
    } else if (asks_done || as <= 0) {
      bad("occupied ask level after an empty one, or empty size", line);
    } else {
      out.asks.push_back(DepthLevel{Price{ap}, as});
    }
    if (bp == kEmptyBid) {
      bids_done = true;
    } else if (bids_done || bs <= 0) {
      bad("occupied bid level after an empty one, or empty size", line);
    } else {
      out.bids.push_back(DepthLevel{Price{bp}, bs});
    }
  }
  if (rest.data() != nullptr && !rest.empty()) bad("orderbook line has too many fields", line);
}

FileInfo describe_file(const std::string& message_path) {
  static const std::regex re(
      R"(^(.*[/\\])?([A-Z.]+)_(\d{4}-\d{2}-\d{2})_(\d+)_(\d+)_message_(\d+)\.csv$)");
  std::smatch m;
  if (!std::regex_match(message_path, m, re)) {
    throw std::runtime_error("not a LOBSTER message file name: " + message_path);
  }
  FileInfo info;
  info.ticker = m[2];
  info.date = m[3];
  info.levels = static_cast<std::size_t>(std::stoul(m[6]));
  const std::size_t at = message_path.rfind("_message_");
  info.orderbook_path = message_path.substr(0, at) + "_orderbook_" + message_path.substr(at + 9);
  return info;
}

DayReader::DayReader(const std::string& message_path, const std::string& orderbook_path,
                     std::size_t levels)
    : msg_(message_path), book_(orderbook_path), levels_(levels) {
  if (!msg_) throw std::runtime_error("cannot open " + message_path);
  if (!book_) throw std::runtime_error("cannot open " + orderbook_path);
}

bool DayReader::next(Message& m, Snapshot& s) {
  const bool have_msg = static_cast<bool>(std::getline(msg_, line_));
  if (have_msg) m = parse_message(line_);
  const bool have_book = static_cast<bool>(std::getline(book_, line_));
  if (have_msg != have_book) {
    throw std::runtime_error("message and orderbook files have different row counts (row " +
                             std::to_string(rows_ + 1) + ")");
  }
  if (!have_msg) return false;
  parse_snapshot(line_, levels_, s);
  ++rows_;
  return true;
}

// ---------------------------------------------------------------- replay

Replay::Replay(std::size_t levels)
    : levels_(levels), book_(&tap_, 1 << 16), next_anon_(kAnonBase), next_taker_(kTakerBase) {}

void Replay::fail(std::string what) {
  if (clean_) {
    ++st_.errors;
    if (st_.first_error == 0) {
      st_.first_error = st_.events;
      st_.first_error_detail = std::move(what);
    }
  }
  clean_ = false;
}

std::optional<OrderId> Replay::anon_at(Side s, Price p) {
  auto& m = anon_[idx(s)];
  const auto it = m.find(p);
  if (it == m.end()) return std::nullopt;
  if (!book_.find(it->second)) {  // used up
    m.erase(it);
    return std::nullopt;
  }
  return it->second;
}

void Replay::add_anon(Side s, Price p, Quantity q, Timestamp ts) {
  const OrderId id = next_anon_++;
  const Result r = book_.submit(Order{id, s, OrderType::PostOnly, p, q, ts});
  if (!r.accepted) {
    fail("could not seed " + std::to_string(q) + " @ " + std::to_string(p.ticks) + ": " +
         std::string(lob::to_string(r.reason)));
    return;
  }
  anon_[idx(s)][p] = id;
}

void Replay::clear_level(Side s, Price p, Timestamp ts) {
  for (OrderId id : book_.queue(s, p)) book_.cancel(id, ts);
  anon_[idx(s)].erase(p);
}

void Replay::reduce(OrderId id, Quantity by, Timestamp ts) {
  const auto o = book_.find(id);
  if (by == o->qty) {
    book_.cancel(id, ts);
  } else {
    book_.modify(id, o->price, o->qty - by, ts);  // size-down: keeps queue priority
  }
}

void Replay::execute(const Message& m, OrderId target) {
  const auto best = book_.best(m.side);
  const std::vector<OrderId> q = book_.queue(m.side, m.price);
  if (best && best->price == m.price && !q.empty() && q.front() == target) {
    tap_.trades.clear();
    const Result r =
        book_.submit(Order{next_taker_++, opposite(m.side), OrderType::IOC, m.price, m.size, m.ns});
    if (r.filled != m.size || tap_.trades.size() != 1 || tap_.trades[0].maker != target) {
      fail("matching engine did not fill the executed order exactly");
    } else {
      ++st_.executions_matched;
    }
  } else {
    ++st_.executions_out_of_order;
    reduce(target, m.size, m.ns);
  }
}

void Replay::seed(const Snapshot& s, Timestamp ts) {
  for (Side side : {Side::Buy, Side::Sell}) {
    const auto& lv = s.side(side);
    for (const DepthLevel& l : lv) {
      add_anon(side, l.price, l.qty, ts);
      ++st_.seeded_levels;
    }
    boundary_[idx(side)] =
        lv.size() == levels_ ? std::optional<Price>(lv.back().price) : std::nullopt;
  }
}

bool Replay::step(const Message& m, const Snapshot& after) {
  ++st_.events;
  ++st_.by_type[static_cast<std::size_t>(m.event)];
  clean_ = true;

  switch (m.event) {
    case Event::Submit: {
      const Result r = book_.submit(Order{m.id, m.side, OrderType::PostOnly, m.price, m.size, m.ns});
      if (!r.accepted) fail("submission refused: " + std::string(lob::to_string(r.reason)));
      break;
    }
    case Event::Cancel:
    case Event::Delete:
    case Event::ExecuteVisible: {
      OrderId target = m.id;
      if (!book_.find(m.id)) {
        const auto a = anon_at(m.side, m.price);
        if (!a) {
          fail("order " + std::to_string(m.id) + " is unknown and no unattributed liquidity rests at " +
               std::to_string(m.price.ticks));
          break;
        }
        target = *a;
        ++st_.unknown_id_events;
      }
      const auto o = book_.find(target);
      if (o->side != m.side || o->price != m.price) {
        fail("order " + std::to_string(m.id) + " rests at " + std::to_string(o->price.ticks) +
             ", message says " + std::to_string(m.price.ticks));
      } else if (m.size <= 0 || m.size > o->qty) {
        fail("size " + std::to_string(m.size) + " exceeds the open quantity " +
             std::to_string(o->qty));
      } else if (m.event == Event::Delete && target == m.id && m.size != o->qty) {
        fail("deleted size " + std::to_string(m.size) + " differs from the open quantity " +
             std::to_string(o->qty));
      } else if (m.event == Event::ExecuteVisible) {
        execute(m, target);
      } else {
        reduce(target, m.size, m.ns);
      }
      break;
    }
    case Event::ExecuteHidden:
    case Event::Cross:
    case Event::Halt:
      break;
  }

  sync_revealed(after, m.ns);
  return compare(after) && clean_;
}

void Replay::sync_revealed(const Snapshot& after, Timestamp ts) {
  for (Side s : {Side::Buy, Side::Sell}) {
    const auto& lv = after.side(s);
    const std::optional<Price> nb =
        lv.size() == levels_ ? std::optional<Price>(lv.back().price) : std::nullopt;
    const std::optional<Price> ob = boundary_[idx(s)];
    boundary_[idx(s)] = nb;
    if (!ob) continue;                         // the whole side was visible: nothing hidden
    if (nb && !better(s, *ob, *nb)) continue;  // the window did not reach further out
    // Newly visible: strictly beyond the old boundary, and within the new one.
    auto in_view = [&](Price p) { return better(s, *ob, p) && (!nb || !better(s, *nb, p)); };
    std::map<Price, std::pair<Quantity, Quantity>> lvls;  // price -> (replayed, snapshot)
    for (const DepthLevel& l : lv) {
      if (in_view(l.price)) lvls[l.price].second = l.qty;
    }
    for (const LevelView& l : book_.depth(s, std::numeric_limits<std::size_t>::max())) {
      if (in_view(l.price)) lvls[l.price].first = l.qty;
      else if (nb && better(s, *nb, l.price)) break;  // depth is best-first: past the window
    }
    for (const auto& [price, q] : lvls) {
      ++st_.revealed_levels;
      if (q.first == q.second) ++st_.revealed_already_right;
      // Rebuild even when the size agrees: which orders sit there is unknowable, and a
      // matching total is no evidence of matching orders.
      clear_level(s, price, ts);
      if (q.second > 0) add_anon(s, price, q.second, ts);
    }
  }
}

bool Replay::compare(const Snapshot& after) {
  bool ok = true;
  for (Side s : {Side::Buy, Side::Sell}) {
    const std::vector<LevelView> ours = book_.depth(s, levels_);
    const auto& ref = after.side(s);
    st_.comparisons += levels_;
    for (std::size_t i = 0; i < levels_ && ok; ++i) {
      const bool have_ours = i < ours.size(), have_ref = i < ref.size();
      if (!have_ours && !have_ref) break;
      if (have_ours && have_ref && ours[i].price == ref[i].price && ours[i].qty == ref[i].qty) {
        continue;
      }
      ok = false;
      if (st_.first_mismatch == 0) {
        std::ostringstream d;
        d << to_string(s) << " level " << i + 1 << ": replayed ";
        if (have_ours) d << ours[i].qty << " @ " << ours[i].price.ticks; else d << "nothing";
        d << ", LOBSTER ";
        if (have_ref) d << ref[i].qty << " @ " << ref[i].price.ticks; else d << "nothing";
        st_.first_mismatch = st_.events;
        st_.first_mismatch_detail = d.str();
      }
    }
  }
  if (!ok) ++st_.mismatches;
  return ok;
}

}  // namespace lob::lobster
