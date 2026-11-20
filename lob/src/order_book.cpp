#include "lob/order_book.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace lob {

namespace {

[[nodiscard]] constexpr bool rests(OrderType t) noexcept {
  return t == OrderType::Limit || t == OrderType::PostOnly;
}

[[nodiscard]] constexpr bool needs_price(OrderType t) noexcept { return t != OrderType::Market; }

// Intrusive FIFO operations on one level.
void push_back(Level& lvl, OrderNode* n) noexcept {
  n->level = &lvl;
  n->prev = lvl.tail;
  n->next = nullptr;
  if (lvl.tail != nullptr) {
    lvl.tail->next = n;
  } else {
    lvl.head = n;
  }
  lvl.tail = n;
  lvl.total += n->qty;
  ++lvl.count;
}

void unlink(Level& lvl, OrderNode* n) noexcept {
  (n->prev != nullptr ? n->prev->next : lvl.head) = n->next;
  (n->next != nullptr ? n->next->prev : lvl.tail) = n->prev;
  lvl.total -= n->qty;
  --lvl.count;
  n->prev = n->next = nullptr;
  n->level = nullptr;
}

std::int64_t checked_tick(std::int64_t tick) {
  if (tick <= 0) throw std::invalid_argument("tick_size must be positive");
  return tick;
}

}  // namespace

template <typename L>
BasicOrderBook<L>::BasicOrderBook(Listener* listener, std::size_t expected_orders,
                                  std::int64_t tick_size)
    : listener_(listener),
      tick_(checked_tick(tick_size)),
      bids_(tick_size),
      asks_(tick_size),
      pool_(expected_orders),
      index_(expected_orders) {}

// ---------------------------------------------------------------- order entry

template <typename L>
Result BasicOrderBook<L>::submit(const Order& o) {
  const Price limit = needs_price(o.type) ? o.price : Price{};
  if (o.qty <= 0) return reject(o.id, o.side, limit, Reason::InvalidQuantity, o.ts);
  if (needs_price(o.type) && !valid_price(o.side, o.price)) {
    return reject(o.id, o.side, limit, Reason::InvalidPrice, o.ts);
  }
  if (index_.find(o.id) != nullptr) {
    return reject(o.id, o.side, limit, Reason::DuplicateOrderId, o.ts);
  }
  if (o.type == OrderType::PostOnly && would_cross(o.side, o.price)) {
    return reject(o.id, o.side, limit, Reason::WouldCross, o.ts);
  }
  if (o.type == OrderType::FOK && available(o.side, o.price, o.qty) < o.qty) {
    return reject(o.id, o.side, limit, Reason::InsufficientLiquidity, o.ts);
  }

  if (rests(o.type)) reserve_one();  // allocate (if ever) before any state changes
  emit(ExecReport{o.id, ExecType::New, Reason::None, o.side, limit, 0, o.qty, o.ts});

  Result r{.accepted = true};
  r.filled = match(o.id, o.side, o.type, limit, o.qty, o.ts);
  const Quantity left = o.qty - r.filled;
  assert(o.type != OrderType::FOK || left == 0);
  if (left > 0) {
    if (rests(o.type)) {
      rest(o.id, o.side, o.type, o.price, left, o.ts);
      r.resting = left;
    } else {
      r.cancelled = left;
      emit(ExecReport{o.id, ExecType::Cancelled, Reason::Unfilled, o.side, limit, left, 0, o.ts});
    }
  }
  return r;
}

template <typename L>
Result BasicOrderBook<L>::cancel(OrderId id, Timestamp ts) {
  OrderNode* n = index_.find(id);
  if (n == nullptr) return reject(id, Side::Buy, Price{}, Reason::UnknownOrder, ts);
  const OrderNode copy = *n;
  remove(n, ts);
  emit(ExecReport{copy.id, ExecType::Cancelled, Reason::UserCancel, copy.side, copy.price,
                  copy.qty, 0, ts});
  return Result{.accepted = true, .cancelled = copy.qty};
}

template <typename L>
Result BasicOrderBook<L>::modify(OrderId id, Price new_price, Quantity new_qty, Timestamp ts) {
  OrderNode* n = index_.find(id);
  if (n == nullptr) return reject(id, Side::Buy, new_price, Reason::UnknownOrder, ts);
  if (new_qty <= 0) return reject(id, n->side, new_price, Reason::InvalidQuantity, ts);
  if (!valid_price(n->side, new_price)) {
    return reject(id, n->side, new_price, Reason::InvalidPrice, ts);
  }

  const Side side = n->side;
  const OrderType type = n->type;

  if (new_price == n->price && new_qty <= n->qty) {
    if (new_qty == n->qty) return Result{.accepted = true, .resting = new_qty};  // no-op
    // Size-down in place: queue position is kept.
    Level& lvl = *n->level;
    lvl.total -= n->qty - new_qty;
    n->qty = new_qty;
    emit(ExecReport{id, ExecType::Replaced, Reason::None, side, new_price, 0, new_qty, ts});
    emit(LevelUpdate{side, lvl.price, lvl.total, lvl.count, ts});
    return Result{.accepted = true, .resting = new_qty};
  }

  // Cancel-replace: loses priority, and may trade if a limit order is re-priced through.
  if (type == OrderType::PostOnly && new_price != n->price && would_cross(side, new_price)) {
    return reject(id, side, new_price, Reason::WouldCross, ts);
  }
  reserve_one();  // allocate (if ever) before any state changes
  emit(ExecReport{id, ExecType::Replaced, Reason::None, side, new_price, 0, new_qty, ts});
  remove(n, ts);
  Result r{.accepted = true};
  r.filled = match(id, side, type, new_price, new_qty, ts);
  if (const Quantity left = new_qty - r.filled; left > 0) {
    rest(id, side, type, new_price, left, ts);
    r.resting = left;
  }
  return r;
}

template <typename L>
Result BasicOrderBook<L>::reject(OrderId id, Side side, Price price, Reason reason,
                                 Timestamp ts) const {
  emit(ExecReport{id, ExecType::Rejected, reason, side, price, 0, 0, ts});
  return Result{.accepted = false, .reason = reason};
}

// ---------------------------------------------------------------- matching

template <typename L>
bool BasicOrderBook<L>::would_cross(Side side, Price limit) const noexcept {
  return on_side(opposite(side), [&](const auto& opp) {
    return !opp.empty() && crosses(side, limit, opp.best().price);
  });
}

template <typename L>
Quantity BasicOrderBook<L>::available(Side side, Price limit, Quantity needed) const noexcept {
  return on_side(opposite(side), [&](const auto& opp) {
    Quantity sum = 0;
    opp.walk([&](const Level& lvl) {
      if (!crosses(side, limit, lvl.price)) return false;
      sum += lvl.total;
      return sum < needed;
    });
    return sum;
  });
}

template <typename L>
Quantity BasicOrderBook<L>::match(OrderId taker, Side side, OrderType type, Price limit,
                                  Quantity qty, Timestamp ts) {
  return on_side(opposite(side), [&](auto& opp) {
    return match_against(opp, taker, side, type, limit, qty, ts);
  });
}

template <typename L>
template <typename Opp>
Quantity BasicOrderBook<L>::match_against(Opp& opp, OrderId taker, Side side, OrderType type,
                                          Price limit, Quantity qty, Timestamp ts) {
  Quantity remaining = qty;
  while (remaining > 0 && !opp.empty()) {
    Level& lvl = opp.best();
    if (type != OrderType::Market && !crosses(side, limit, lvl.price)) break;

    while (remaining > 0 && lvl.head != nullptr) {
      OrderNode* maker = lvl.head;
      const Quantity q = std::min(remaining, maker->qty);
      maker->qty -= q;
      lvl.total -= q;
      remaining -= q;
      emit(Trade{taker, maker->id, side, lvl.price, q, ts});
      emit(ExecReport{maker->id, ExecType::Fill, Reason::None, maker->side, lvl.price, q,
                      maker->qty, ts});
      emit(ExecReport{taker, ExecType::Fill, Reason::None, side, lvl.price, q, remaining, ts});
      if (maker->qty == 0) {
        unlink(lvl, maker);  // qty already 0, so the level total is unaffected
        index_.erase(maker->id);
        pool_.release(maker);
      }
    }

    // One update per level touched, after all fills at that level.
    emit(LevelUpdate{opposite(side), lvl.price, lvl.total, lvl.count, ts});
    if (lvl.count == 0) opp.erase(lvl.price);  // `lvl` dangles after this line
  }
  return qty - remaining;
}

// ---------------------------------------------------------------- resting orders

template <typename L>
void BasicOrderBook<L>::reserve_one() {
  pool_.ensure_free(1);
  index_.reserve(index_.size() + 1);
}

template <typename L>
void BasicOrderBook<L>::rest(OrderId id, Side side, OrderType type, Price price, Quantity qty,
                             Timestamp ts) {
  OrderNode* n = pool_.allocate();  // free slot guaranteed by reserve_one()
  Level* lvl = nullptr;
  try {
    lvl = on_side(side, [&](auto& s) { return &s.get_or_create(price); });
  } catch (...) {
    pool_.release(n);
    throw;
  }
  *n = OrderNode{.id = id, .price = price, .qty = qty, .ts = ts, .seq = ++next_seq_,
                 .side = side, .type = type};
  push_back(*lvl, n);
  index_.insert(id, n);  // capacity guaranteed by reserve_one()
  emit(LevelUpdate{side, price, lvl->total, lvl->count, ts});
}

template <typename L>
void BasicOrderBook<L>::remove(OrderNode* n, Timestamp ts) {
  Level& lvl = *n->level;
  const Side side = n->side;
  const Price price = lvl.price;
  unlink(lvl, n);
  index_.erase(n->id);
  pool_.release(n);
  emit(LevelUpdate{side, price, lvl.total, lvl.count, ts});
  if (lvl.count == 0) on_side(side, [&](auto& s) { s.erase(price); });
}

// ---------------------------------------------------------------- views

template <typename L>
std::optional<Price> BasicOrderBook<L>::best_bid() const noexcept {
  if (bids_.empty()) return std::nullopt;
  return bids_.best().price;
}

template <typename L>
std::optional<Price> BasicOrderBook<L>::best_ask() const noexcept {
  if (asks_.empty()) return std::nullopt;
  return asks_.best().price;
}

template <typename L>
std::optional<LevelView> BasicOrderBook<L>::best(Side side) const noexcept {
  return on_side(side, [](const auto& s) -> std::optional<LevelView> {
    if (s.empty()) return std::nullopt;
    const Level& l = s.best();
    return LevelView{l.price, l.total, l.count};
  });
}

template <typename L>
std::vector<LevelView> BasicOrderBook<L>::depth(Side side, std::size_t max_levels) const {
  return on_side(side, [&](const auto& s) {
    std::vector<LevelView> out;
    out.reserve(std::min(max_levels, s.size()));
    if (max_levels == 0) return out;
    s.walk([&](const Level& l) {
      out.push_back(LevelView{l.price, l.total, l.count});
      return out.size() < max_levels;
    });
    return out;
  });
}

template <typename L>
std::vector<OrderId> BasicOrderBook<L>::queue(Side side, Price price) const {
  return on_side(side, [&](const auto& s) {
    std::vector<OrderId> ids;
    if (const Level* l = s.find(price)) {
      ids.reserve(l->count);
      for (const OrderNode* n = l->head; n != nullptr; n = n->next) ids.push_back(n->id);
    }
    return ids;
  });
}

template <typename L>
std::vector<OrderView> BasicOrderBook<L>::orders(Side side) const {
  return on_side(side, [](const auto& s) {
    std::vector<OrderView> out;
    s.walk([&](const Level& lvl) {
      for (const OrderNode* n = lvl.head; n != nullptr; n = n->next) {
        out.push_back(OrderView{n->id, n->side, n->type, n->price, n->qty, n->ts});
      }
      return true;
    });
    return out;
  });
}

template <typename L>
std::optional<OrderView> BasicOrderBook<L>::find(OrderId id) const noexcept {
  const OrderNode* n = index_.find(id);
  if (n == nullptr) return std::nullopt;
  return OrderView{n->id, n->side, n->type, n->price, n->qty, n->ts};
}

template <typename L>
std::size_t BasicOrderBook<L>::level_count(Side side) const noexcept {
  return side == Side::Buy ? bids_.size() : asks_.size();
}

template <typename L>
Quantity BasicOrderBook<L>::resting_qty(Side side) const noexcept {
  return on_side(side, [](const auto& s) {
    Quantity sum = 0;
    s.walk([&](const Level& lvl) {
      sum += lvl.total;
      return true;
    });
    return sum;
  });
}

// ---------------------------------------------------------------- self-check

template <typename L>
std::string BasicOrderBook<L>::validate() const {
  std::ostringstream err;
  std::size_t nodes = 0;

  auto check_side = [&](const auto& s, Side side) {
    std::size_t levels = 0;
    const Level* prev_lvl = nullptr;
    s.walk([&](const Level& lvl) {
      ++levels;
      const Price price = lvl.price;
      if (prev_lvl != nullptr && !better(side, prev_lvl->price, price)) {
        err << to_string(side) << " levels out of priority order at " << price.ticks;
      } else if (s.find(price) != &lvl) {
        err << to_string(side) << " level " << price.ticks << " not found at its own price";
      } else if (lvl.count == 0 || lvl.head == nullptr) {
        err << to_string(side) << " level " << price.ticks << " is empty but present";
      }
      if (err.tellp() > 0) return false;
      prev_lvl = &lvl;
      Quantity total = 0;
      std::uint32_t count = 0;
      const OrderNode* prev = nullptr;
      for (const OrderNode* n = lvl.head; n != nullptr; prev = n, n = n->next) {
        ++count;
        ++nodes;
        total += n->qty;
        if (n->prev != prev) err << "order " << n->id << ": broken prev link";
        else if (n->level != &lvl) err << "order " << n->id << ": wrong level pointer";
        else if (n->price != price) err << "order " << n->id << ": price differs from level";
        else if (n->side != side) err << "order " << n->id << ": on the wrong side";
        else if (n->qty <= 0) err << "order " << n->id << ": non-positive open quantity";
        else if (prev != nullptr && prev->seq >= n->seq)
          err << "order " << n->id << ": FIFO violated at " << price.ticks;
        else if (index_.find(n->id) != n) err << "order " << n->id << ": index mismatch";
        if (err.tellp() > 0) return false;
      }
      if (lvl.tail != prev) err << to_string(side) << " level " << price.ticks << ": bad tail";
      else if (total != lvl.total)
        err << to_string(side) << " level " << price.ticks << ": total " << lvl.total
            << " != sum of orders " << total;
      else if (count != lvl.count)
        err << to_string(side) << " level " << price.ticks << ": count " << lvl.count
            << " != " << count;
      return err.tellp() == 0;
    });
    if (err.tellp() == 0 && levels != s.size()) {
      err << to_string(side) << " side reports " << s.size() << " levels, walked " << levels;
    }
  };

  check_side(bids_, Side::Buy);
  if (err.tellp() == 0) check_side(asks_, Side::Sell);
  if (err.tellp() == 0 && !bids_.empty() && !asks_.empty() &&
      bids_.best().price >= asks_.best().price) {
    err << "crossed book: bid " << bids_.best().price.ticks << " >= ask "
        << asks_.best().price.ticks;
  }
  if (err.tellp() == 0 && nodes != index_.size()) {
    err << "index holds " << index_.size() << " ids but book holds " << nodes << " orders";
  }
  if (err.tellp() == 0 && nodes != pool_.in_use()) {
    err << "pool has " << pool_.in_use() << " slots in use but book holds " << nodes;
  }
  return err.str();
}

template class BasicOrderBook<MapLevels>;
template class BasicOrderBook<ArrayLevels>;

}  // namespace lob
