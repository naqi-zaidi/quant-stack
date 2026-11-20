// LOBSTER parsing and replay on hand-built days. These run in CI; the real-data gate
// (lob_replay on the AAPL sample) needs the sample file locally.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include "lob/lobster.hpp"

namespace lob::lobster {
namespace {

using Levels = std::initializer_list<DepthLevel>;

Snapshot snap(Levels asks, Levels bids) {
  Snapshot s;
  s.asks.assign(asks);
  s.bids.assign(bids);
  return s;
}

Message msg(Event e, OrderId id, Quantity size, std::int64_t px, int dir) {
  return Message{0, e, id, size, Price{px}, dir == 1 ? Side::Buy : Side::Sell};
}

DepthLevel lv(std::int64_t px, Quantity q) { return DepthLevel{Price{px}, q}; }

// ---------------------------------------------------------------- parsing

TEST(LobsterParse, MessageLine) {
  const Message m = parse_message("34200.017459617,1,16113575,18,5853300,1");
  EXPECT_EQ(m.ns, 34'200'017'459'617u);  // exact nanoseconds, no floating point
  EXPECT_EQ(m.event, Event::Submit);
  EXPECT_EQ(m.id, 16113575u);
  EXPECT_EQ(m.size, 18);
  EXPECT_EQ(m.price, Price{5853300});
  EXPECT_EQ(m.side, Side::Buy);

  const Message n = parse_message("34200.5,3,1,2,3,-1\r");
  EXPECT_EQ(n.ns, 34'200'500'000'000u);
  EXPECT_EQ(n.side, Side::Sell);
  EXPECT_EQ(parse_message("34201,4,7,1,5,1,0,0").ns, 34'201'000'000'000u);  // extra columns
}

TEST(LobsterParse, MalformedMessagesThrow) {
  for (const char* bad : {"34200.1,1,1,1,1", "34200.1,9,1,1,1,1", "34200.1,1,1,1,1,0",
                          "abc,1,1,1,1,1", "34200.1,1,x,1,1,1", "34200.1x,1,1,1,1,1"}) {
    EXPECT_THROW((void)parse_message(bad), std::runtime_error) << bad;
  }
}

TEST(LobsterParse, SnapshotLineSkipsEmptyLevels) {
  Snapshot s;
  parse_snapshot("5853500,50,5853300,100,9999999999,0,-9999999999,0", 2, s);
  EXPECT_EQ(s.asks, (std::vector<DepthLevel>{lv(5853500, 50)}));
  EXPECT_EQ(s.bids, (std::vector<DepthLevel>{lv(5853300, 100)}));
  parse_snapshot("9999999999,0,5853300,100,9999999999,0,5853200,20", 2, s);
  EXPECT_TRUE(s.asks.empty());
  EXPECT_EQ(s.bids.size(), 2u);
}

TEST(LobsterParse, MalformedSnapshotsThrow) {
  Snapshot s;
  EXPECT_THROW(parse_snapshot("1,1,1,1,1,1,1", 2, s), std::runtime_error);  // too few
  EXPECT_THROW(parse_snapshot("1,1,1,1,1,1,1,1,1", 2, s), std::runtime_error);  // too many
  EXPECT_THROW(parse_snapshot("9999999999,0,5,1,7,3,4,1", 2, s), std::runtime_error);  // gap
}

TEST(LobsterParse, FileNameGivesTickerLevelsAndOrderbookPath) {
  const FileInfo f = describe_file("data/AAPL_2012-06-21_34200000_57600000_message_10.csv");
  EXPECT_EQ(f.ticker, "AAPL");
  EXPECT_EQ(f.date, "2012-06-21");
  EXPECT_EQ(f.levels, 10u);
  EXPECT_EQ(f.orderbook_path, "data/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv");
  EXPECT_THROW((void)describe_file("data/AAPL_orderbook_10.csv"), std::runtime_error);
}

// ---------------------------------------------------------------- replay

// A five-event day with a 2-level book (the same day the Python loader tests use):
//   0 buy 100 @ 585.33 (seeds the book)   1 sell 50 @ 585.35   2 buy 20 @ 585.32
//   3 10 of order 2 executes               4 order 3 deleted
struct Day {
  std::vector<Message> msgs{
      msg(Event::Submit, 1, 100, 5853300, 1),         msg(Event::Submit, 2, 50, 5853500, -1),
      msg(Event::Submit, 3, 20, 5853200, 1),          msg(Event::ExecuteVisible, 2, 10, 5853500, -1),
      msg(Event::Delete, 3, 20, 5853200, 1),
  };
  std::vector<Snapshot> books{
      snap({}, {lv(5853300, 100)}),
      snap({lv(5853500, 50)}, {lv(5853300, 100)}),
      snap({lv(5853500, 50)}, {lv(5853300, 100), lv(5853200, 20)}),
      snap({lv(5853500, 40)}, {lv(5853300, 100), lv(5853200, 20)}),
      snap({lv(5853500, 40)}, {lv(5853300, 100)}),
  };
};

ReplayStats run(const Day& d, std::size_t levels = 2) {
  Replay rp(levels);
  rp.seed(d.books[0]);
  for (std::size_t i = 1; i < d.msgs.size(); ++i) rp.step(d.msgs[i], d.books[i]);
  return rp.stats();
}

TEST(LobsterReplay, ConsistentDayMatchesOnEveryEvent) {
  const ReplayStats st = run(Day{});
  EXPECT_EQ(st.events, 4u);
  EXPECT_EQ(st.mismatches, 0u) << st.first_mismatch_detail;
  EXPECT_EQ(st.errors, 0u) << st.first_error_detail;
  EXPECT_EQ(st.seeded_levels, 1u);
  EXPECT_EQ(st.executions_matched, 1u);  // the execution went through the matching engine
  EXPECT_EQ(st.comparisons, 4u * 2u * 2u);
}

TEST(LobsterReplay, CorruptedSnapshotIsCaughtAtThatEvent) {
  Day d;
  d.books[3].asks[0].qty = 41;  // LOBSTER says 40 after the execution
  const ReplayStats st = run(d);
  EXPECT_EQ(st.first_mismatch, 3u);
  EXPECT_NE(st.first_mismatch_detail.find("Sell level 1"), std::string::npos);
}

TEST(LobsterReplay, MissingMessageIsCaughtAtThatEvent) {
  Day d;
  d.msgs[2].event = Event::Halt;  // drop "buy 20 @ 585.32"
  const ReplayStats st = run(d);
  EXPECT_EQ(st.first_mismatch, 2u);
}

TEST(LobsterReplay, PreOpenOrdersDrawOnSeededLiquidity) {
  Replay rp(2);
  rp.seed(snap({}, {lv(1000, 100)}));
  // Order 999 rested before the open, so the file never submitted it.
  EXPECT_TRUE(rp.step(msg(Event::Cancel, 999, 30, 1000, 1), snap({}, {lv(1000, 70)})));
  EXPECT_TRUE(rp.step(msg(Event::ExecuteVisible, 998, 20, 1000, 1), snap({}, {lv(1000, 50)})));
  EXPECT_TRUE(rp.step(msg(Event::Delete, 997, 50, 1000, 1), snap({}, {})));
  EXPECT_EQ(rp.stats().unknown_id_events, 3u);
  EXPECT_EQ(rp.stats().executions_matched, 1u);
  EXPECT_EQ(rp.book().order_count(), 0u);
}

TEST(LobsterReplay, ExecutionOfAnOrderNotFirstInQueueIsCountedAndFollowed) {
  Replay rp(2);
  rp.seed(snap({}, {}));
  rp.step(msg(Event::Submit, 1, 10, 2000, -1), snap({lv(2000, 10)}, {}));
  rp.step(msg(Event::Submit, 2, 10, 2000, -1), snap({lv(2000, 20)}, {}));
  // Nasdaq filled order 2 although order 1 was ahead of it.
  EXPECT_TRUE(rp.step(msg(Event::ExecuteVisible, 2, 10, 2000, -1), snap({lv(2000, 10)}, {})));
  EXPECT_EQ(rp.stats().executions_out_of_order, 1u);
  EXPECT_EQ(rp.stats().executions_matched, 0u);
  ASSERT_TRUE(rp.book().find(1));  // book state follows what happened: order 1 still rests
  EXPECT_FALSE(rp.book().find(2));
}

TEST(LobsterReplay, LevelScrollingIntoViewIsRebuiltFromTheSnapshot) {
  Replay rp(2);
  rp.seed(snap({lv(101, 10), lv(102, 20)}, {}));  // two levels shown: more may lie beyond
  // The best ask goes; 103 scrolls into view with 30 the replay has never seen.
  EXPECT_TRUE(rp.step(msg(Event::Delete, 50, 10, 101, -1), snap({lv(102, 20), lv(103, 30)}, {})));
  EXPECT_EQ(rp.stats().revealed_levels, 1u);
  EXPECT_EQ(rp.stats().revealed_already_right, 0u);
  // Later events on the unseen orders draw on that rebuilt block.
  EXPECT_TRUE(rp.step(msg(Event::ExecuteVisible, 51, 20, 102, -1), snap({lv(103, 30)}, {})));
  EXPECT_TRUE(rp.step(msg(Event::Delete, 52, 30, 103, -1), snap({}, {})));
  EXPECT_EQ(rp.stats().errors, 0u) << rp.stats().first_error_detail;
}

TEST(LobsterReplay, StaleOutOfViewLevelIsReplacedWhenItReturns) {
  Replay rp(1);
  rp.seed(snap({lv(102, 7)}, {}));
  rp.step(msg(Event::Submit, 1, 10, 101, -1), snap({lv(101, 10)}, {}));  // 102 drops out of view
  // While out of view, 102 changed from 7 to 4 without a message. When 101 goes, 102 is
  // newly visible again and is rebuilt from the snapshot.
  EXPECT_TRUE(rp.step(msg(Event::Delete, 1, 10, 101, -1), snap({lv(102, 4)}, {})));
  EXPECT_EQ(rp.stats().revealed_levels, 1u);
  EXPECT_EQ(rp.stats().mismatches, 0u);
}

TEST(LobsterReplay, AlreadyVisibleLevelsAreNeverResynced) {
  Replay rp(2);
  rp.seed(snap({lv(101, 10), lv(102, 20)}, {}));
  // The snapshot claims 102 changed, but 102 was visible and no message touched it: that is
  // a mismatch, not something to paper over.
  EXPECT_FALSE(rp.step(msg(Event::ExecuteHidden, 0, 5, 101, -1), snap({lv(101, 10), lv(102, 25)}, {})));
  EXPECT_EQ(rp.stats().revealed_levels, 0u);
  EXPECT_EQ(rp.stats().first_mismatch, 1u);
}

TEST(LobsterReplay, InconsistentEventsAreErrors) {
  {
    Replay rp(2);
    rp.seed(snap({lv(101, 10)}, {}));
    EXPECT_FALSE(rp.step(msg(Event::Submit, 1, 5, 101, 1), snap({lv(101, 10)}, {lv(101, 5)})));
    EXPECT_NE(rp.stats().first_error_detail.find("WouldCross"), std::string::npos);
  }
  {
    Replay rp(2);
    rp.seed(snap({lv(101, 10)}, {}));
    EXPECT_FALSE(rp.step(msg(Event::Delete, 7, 5, 105, -1), snap({lv(101, 10)}, {})));
    EXPECT_NE(rp.stats().first_error_detail.find("unknown"), std::string::npos);
  }
  {
    Replay rp(2);
    rp.seed(snap({}, {}));
    rp.step(msg(Event::Submit, 1, 10, 101, -1), snap({lv(101, 10)}, {}));
    EXPECT_FALSE(rp.step(msg(Event::Delete, 1, 6, 101, -1), snap({}, {})));
    EXPECT_NE(rp.stats().first_error_detail.find("differs"), std::string::npos);
  }
  {
    Replay rp(2);
    rp.seed(snap({}, {}));
    rp.step(msg(Event::Submit, 1, 10, 101, -1), snap({lv(101, 10)}, {}));
    EXPECT_FALSE(rp.step(msg(Event::ExecuteVisible, 1, 11, 101, -1), snap({}, {})));
    EXPECT_NE(rp.stats().first_error_detail.find("exceeds"), std::string::npos);
  }
}

// ---------------------------------------------------------------- reading files

class LobsterFiles : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() /
           ("lob_lobster_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
            "_" + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(dir_);
    msg_ = (dir_ / "TEST_2012-06-21_34200000_57600000_message_2.csv").string();
    book_ = (dir_ / "TEST_2012-06-21_34200000_57600000_orderbook_2.csv").string();
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }

  void write(const std::string& path, const std::vector<std::string>& lines) {
    std::ofstream f(path);
    for (const auto& l : lines) f << l << "\n";
  }

  const std::vector<std::string> msgs_{
      "34200.001,1,1,100,5853300,1",     "34200.002,1,2,50,5853500,-1",
      "34200.003,1,3,20,5853200,1",      "34200.004,4,2,10,5853500,-1",
      "34200.005,3,3,20,5853200,1",
  };
  const std::vector<std::string> books_{
      "9999999999,0,5853300,100,9999999999,0,-9999999999,0",
      "5853500,50,5853300,100,9999999999,0,-9999999999,0",
      "5853500,50,5853300,100,9999999999,0,5853200,20",
      "5853500,40,5853300,100,9999999999,0,5853200,20",
      "5853500,40,5853300,100,9999999999,0,-9999999999,0",
  };
  std::filesystem::path dir_;
  std::string msg_, book_;
};

TEST_F(LobsterFiles, ReadsBothFilesInLockstepAndReplaysCleanly) {
  write(msg_, msgs_);
  write(book_, books_);
  const FileInfo info = describe_file(msg_);
  EXPECT_EQ(info.orderbook_path, book_);
  DayReader rd(msg_, info.orderbook_path, info.levels);
  Message m;
  Snapshot s;
  ASSERT_TRUE(rd.next(m, s));
  Replay rp(info.levels);
  rp.seed(s, m.ns);
  while (rd.next(m, s)) EXPECT_TRUE(rp.step(m, s));
  EXPECT_EQ(rd.rows(), 5u);
  EXPECT_EQ(rp.stats().mismatches, 0u);
}

TEST_F(LobsterFiles, RowCountMismatchThrows) {
  write(msg_, msgs_);
  write(book_, std::vector<std::string>(books_.begin(), books_.end() - 1));
  DayReader rd(msg_, book_, 2);
  Message m;
  Snapshot s;
  auto drain = [&] {
    while (rd.next(m, s)) {
    }
  };
  EXPECT_THROW(drain(), std::runtime_error);
}

TEST_F(LobsterFiles, MissingFileThrows) {
  EXPECT_THROW((void)DayReader(msg_, book_, 2), std::runtime_error);
}

}  // namespace
}  // namespace lob::lobster
