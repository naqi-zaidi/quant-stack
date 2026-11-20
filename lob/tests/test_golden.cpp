// Snapshot ("golden") tests: each scenario prints every call, its result, every event the
// book emitted (in arrival order) and the book itself, and the text must equal
// lob/tests/golden/<name>.txt byte for byte. Printing the whole state catches changes no
// assertion anticipated.
//
// To accept an intended change, regenerate and review the diff in the commit:
//   UPDATE_GOLDEN=1 ./build/release/lob/tests/lob_tests --gtest_filter='Golden*'

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "lob/lobster.hpp"
#include "lob/order_book.hpp"
#include "lob/text.hpp"

#ifndef LOB_GOLDEN_DIR
#error "LOB_GOLDEN_DIR must point at lob/tests/golden"
#endif

namespace lob {
namespace {

bool updating() {
#ifdef _MSC_VER
  char* v = nullptr;
  std::size_t n = 0;
  const bool on = _dupenv_s(&v, &n, "UPDATE_GOLDEN") == 0 && v != nullptr && std::string(v) == "1";
  std::free(v);
  return on;
#else
  const char* v = std::getenv("UPDATE_GOLDEN");
  return v != nullptr && std::string(v) == "1";
#endif
}

void expect_golden(const std::string& name, const std::string& actual) {
  const std::filesystem::path path = std::filesystem::path(LOB_GOLDEN_DIR) / (name + ".txt");
  if (updating()) {
    std::ofstream(path, std::ios::binary) << actual;
    return;
  }
  std::ifstream in(path, std::ios::binary);
  ASSERT_TRUE(in) << "missing golden file " << path << "; generate it with UPDATE_GOLDEN=1";
  std::ostringstream expected;
  expected << in.rdbuf();
  EXPECT_EQ(expected.str(), actual)
      << "golden " << name << " changed. If intended, regenerate with UPDATE_GOLDEN=1 and "
         "review the diff in the commit.";
}

std::string result_text(const Result& r) {
  std::ostringstream os;
  if (!r.accepted) {
    os << "rejected (" << to_string(r.reason) << ")";
  } else {
    os << "accepted: filled " << r.filled << ", resting " << r.resting << ", cancelled "
       << r.cancelled;
  }
  return os.str();
}

// Drives a book through a scenario and transcribes it. Each call gets the next timestamp.
template <typename Book>
class Script {
 public:
  explicit Script(const std::string& title) { out_ << "# " << title << "\n"; }

  Script& submit(OrderId id, Side s, OrderType t, std::int64_t px, Quantity q) {
    std::ostringstream c;
    c << "submit #" << id << " " << to_string(s) << " " << to_string(t) << " " << q;
    if (t != OrderType::Market) c << " @ " << px;
    return record(c.str(), book_.submit(Order{id, s, t, Price{px}, q, ++ts_}));
  }
  Script& cancel(OrderId id) {
    return record("cancel #" + std::to_string(id), book_.cancel(id, ++ts_));
  }
  Script& modify(OrderId id, std::int64_t px, Quantity q) {
    std::ostringstream c;
    c << "modify #" << id << " to " << q << " @ " << px;
    return record(c.str(), book_.modify(id, Price{px}, q, ++ts_));
  }
  // The standard small book, built quietly:
  //   asks 101: [1:10, 2:20]   102: [3:30]   103: [4:40]
  //   bids  99: [5:10, 6:20]    98: [7:30]
  Script& seed() {
    for (const auto& o : {Order{1, Side::Sell, OrderType::Limit, Price{101}, 10},
                          Order{2, Side::Sell, OrderType::Limit, Price{101}, 20},
                          Order{3, Side::Sell, OrderType::Limit, Price{102}, 30},
                          Order{4, Side::Sell, OrderType::Limit, Price{103}, 40},
                          Order{5, Side::Buy, OrderType::Limit, Price{99}, 10},
                          Order{6, Side::Buy, OrderType::Limit, Price{99}, 20},
                          Order{7, Side::Buy, OrderType::Limit, Price{98}, 30}}) {
      book_.submit(o);
    }
    log_.take();
    out_ << "\nseeded book\n" << to_text(book_);
    return *this;
  }
  Script& show() {
    out_ << "\nbook\n" << to_text(book_);
    return *this;
  }
  [[nodiscard]] std::string text() const { return out_.str(); }

 private:
  Script& record(const std::string& call, const Result& r) {
    out_ << "\n> " << call << "\n  = " << result_text(r) << "\n" << log_.take("  ");
    return *this;
  }

  TextLog log_;
  Book book_{&log_, 64};
  std::ostringstream out_;
  Timestamp ts_{0};
};

constexpr Side B = Side::Buy;
constexpr Side S = Side::Sell;
using T = OrderType;

template <typename Book>
std::string ladder() {
  return Script<Book>("the standard book, order by order")
      .submit(1, S, T::Limit, 101, 10).submit(2, S, T::Limit, 101, 20)
      .submit(3, S, T::Limit, 102, 30).submit(4, S, T::Limit, 103, 40)
      .submit(5, B, T::Limit, 99, 10).submit(6, B, T::Limit, 99, 20)
      .submit(7, B, T::Limit, 98, 30).show().text();
}

template <typename Book>
std::string limit_sweeps_and_rests() {
  return Script<Book>("a buy limit sweeps two levels in price-time order and rests the rest")
      .seed().submit(10, B, T::Limit, 102, 70).show().text();
}

template <typename Book>
std::string market_and_ioc() {
  return Script<Book>("market and IOC orders cancel what they cannot fill")
      .seed().submit(10, S, T::Market, 0, 100).submit(11, B, T::IOC, 102, 100).show().text();
}

template <typename Book>
std::string fill_or_kill() {
  return Script<Book>("fill-or-kill: killed one share short, filled at exactly enough")
      .seed().submit(10, B, T::FOK, 102, 61).submit(11, B, T::FOK, 102, 60).show().text();
}

template <typename Book>
std::string post_only() {
  return Script<Book>("post-only: rejected at and through the touch, rests when passive")
      .seed().submit(10, B, T::PostOnly, 101, 5).submit(11, B, T::PostOnly, 110, 5)
      .submit(12, S, T::PostOnly, 100, 5).show().text();
}

template <typename Book>
std::string cancels() {
  return Script<Book>("cancel from the middle of a queue, an emptied level, an unknown id")
      .seed().submit(8, B, T::Limit, 99, 5).cancel(6).cancel(3).cancel(999).cancel(6).show()
      .text();
}

template <typename Book>
std::string modify_priority() {
  return Script<Book>("modify: size-down keeps priority, size-up and re-price lose it, a re-price "
                      "through the spread trades")
      .submit(1, S, T::Limit, 101, 10).submit(2, S, T::Limit, 101, 10)
      .submit(3, S, T::Limit, 101, 10).submit(4, B, T::Limit, 99, 20)
      .modify(1, 101, 4).modify(2, 101, 15).modify(3, 102, 10).show()
      .modify(4, 101, 20).show().text();
}

template <typename Book>
std::string rejections() {
  return Script<Book>("every rejection leaves the book untouched")
      .seed().submit(10, B, T::Limit, 99, 0).submit(11, B, T::IOC, 0, 5)
      .submit(3, B, T::Limit, 50, 5).cancel(999).modify(999, 101, 5).modify(1, 101, 0)
      .modify(1, -5, 5).show().text();
}

template <typename Book>
std::string partial_fills_keep_place() {
  return Script<Book>("a partly filled maker keeps its place at the head of the queue")
      .submit(1, S, T::Limit, 101, 10).submit(2, S, T::Limit, 101, 10)
      .submit(10, B, T::IOC, 101, 4).show().submit(11, B, T::IOC, 101, 8).show().text();
}

// The five-event LOBSTER day from test_lobster.cpp, replayed with every book event shown.
template <typename R>
std::string lobster_day() {
  using lobster::Event;
  using lobster::Message;
  using lobster::Snapshot;
  auto snap = [](std::initializer_list<lobster::DepthLevel> a,
                 std::initializer_list<lobster::DepthLevel> b) {
    Snapshot s;
    s.asks.assign(a);
    s.bids.assign(b);
    return s;
  };
  const std::vector<Message> msgs{
      {1, Event::Submit, 1, 100, Price{5853300}, B},
      {2, Event::Submit, 2, 50, Price{5853500}, S},
      {3, Event::Submit, 3, 20, Price{5853200}, B},
      {4, Event::ExecuteVisible, 2, 10, Price{5853500}, S},
      {5, Event::Delete, 3, 20, Price{5853200}, B},
  };
  const std::vector<Snapshot> books{
      snap({}, {{Price{5853300}, 100}}),
      snap({{Price{5853500}, 50}}, {{Price{5853300}, 100}}),
      snap({{Price{5853500}, 50}}, {{Price{5853300}, 100}, {Price{5853200}, 20}}),
      snap({{Price{5853500}, 40}}, {{Price{5853300}, 100}, {Price{5853200}, 20}}),
      snap({{Price{5853500}, 40}}, {{Price{5853300}, 100}}),
  };
  TextLog log;
  R rp(2);
  rp.set_listener(&log);
  std::ostringstream out;
  out << "# LOBSTER replay of a five-event day (2 levels)\n"
      << "# liquidity the replay seeds itself gets ids from 2^62 = 4611686018427387904 up\n\n"
      << "> seed from the first snapshot\n";
  rp.seed(books[0], msgs[0].ns);
  out << log.take("  ") << to_text(rp.book());
  for (std::size_t i = 1; i < msgs.size(); ++i) {
    const bool ok = rp.step(msgs[i], books[i]);
    out << "\n> event " << i << ": " << lobster::to_string(msgs[i].event) << " #" << msgs[i].id
        << " " << msgs[i].size << " @ " << msgs[i].price.ticks << (ok ? "" : "  (MISMATCH)")
        << "\n" << log.take("  ") << to_text(rp.book());
  }
  return out.str();
}

TEST(Golden, Ladder) { expect_golden("ladder", ladder<OrderBook>()); }
TEST(Golden, LimitSweepsAndRests) {
  expect_golden("limit_sweeps_and_rests", limit_sweeps_and_rests<OrderBook>());
}
TEST(Golden, MarketAndIoc) { expect_golden("market_and_ioc", market_and_ioc<OrderBook>()); }
TEST(Golden, FillOrKill) { expect_golden("fill_or_kill", fill_or_kill<OrderBook>()); }
TEST(Golden, PostOnly) { expect_golden("post_only", post_only<OrderBook>()); }
TEST(Golden, Cancels) { expect_golden("cancels", cancels<OrderBook>()); }
TEST(Golden, ModifyPriority) { expect_golden("modify_priority", modify_priority<OrderBook>()); }
TEST(Golden, Rejections) { expect_golden("rejections", rejections<OrderBook>()); }
TEST(Golden, PartialFillsKeepPlace) {
  expect_golden("partial_fills_keep_place", partial_fills_keep_place<OrderBook>());
}
TEST(Golden, LobsterDay) { expect_golden("lobster_day", lobster_day<lobster::Replay>()); }

}  // namespace
}  // namespace lob
