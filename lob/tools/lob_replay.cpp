// Phase 2 step 3 gate: replay a LOBSTER day through the matching engine and compare the
// rebuilt book with LOBSTER's after every event.
//
//   lob_replay data/AAPL_2012-06-21_34200000_57600000_message_10.csv [drop_checks=12]
//
// Prints a markdown report. Exits non-zero unless every event applies cleanly, every visible
// level matches after every event, and every drop-one-message trial is detected. A trial
// deletes one real message (submission, cancel, deletion or visible execution) and requires
// the replay to flag a problem at exactly that event: evidence the comparison is sharp enough
// to notice any single missing event.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

#include "lob/digest.hpp"
#include "lob/lobster.hpp"

using namespace lob;
using namespace lob::lobster;

namespace {

struct Run {
  ReplayStats stats;
  std::uint64_t digest{0};  // FNV-1a 64 of every book event during the replay
  std::size_t digest_events{0};
  std::vector<std::uint8_t> types;  // event type of each replayed event (main run only)
  double seconds{0};
};

// Replays the day. `drop` (1-based event number, 0 = none) replaces that message with a no-op
// while still comparing against its snapshot. `stop_early` ends at the first problem.
Run replay(const std::string& msg_path, const FileInfo& info, std::size_t drop, bool stop_early,
           bool keep_types) {
  const auto t0 = std::chrono::steady_clock::now();
  DayReader rd(msg_path, info.orderbook_path, info.levels);
  Message m;
  Snapshot s;
  if (!rd.next(m, s)) throw std::runtime_error("empty file");
  Replay rp(info.levels);
  EventDigest dg;
  rp.set_listener(&dg);
  rp.seed(s, m.ns);
  Run run;
  std::size_t k = 0;
  while (rd.next(m, s)) {
    ++k;
    if (keep_types) run.types.push_back(static_cast<std::uint8_t>(m.event));
    if (k == drop) m.event = Event::Halt;  // the dropped message: applies nothing
    const bool ok = rp.step(m, s);
    if (stop_early && !ok) break;
  }
  run.stats = rp.stats();
  run.digest = dg.value();
  run.digest_events = dg.events();
  run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return run;
}

std::string pct(std::size_t a, std::size_t b) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << (b == 0 ? 0.0 : 100.0 * static_cast<double>(a) / static_cast<double>(b)) << "%";
  return os.str();
}

// splitmix64, so the chosen trials are the same on every platform.
std::uint64_t mix(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    std::cerr << "usage: lob_replay <TICKER_DATE_START_END_message_LEVELS.csv> [drop_checks=12]\n";
    return 2;
  }
  const std::string msg_path = argv[1];
  const std::size_t trials = argc > 2 ? std::stoul(argv[2]) : 12;
  const FileInfo info = describe_file(msg_path);

  const Run main_run = replay(msg_path, info, 0, false, true);
  const ReplayStats& st = main_run.stats;
  bool ok = st.mismatches == 0 && st.errors == 0;

  std::cout << "## LOBSTER replay: " << info.ticker << " " << info.date << ", " << info.levels
            << " levels\n\n"
            << "| Measure | Value |\n|---|---:|\n"
            << "| Events replayed (after the seeding snapshot) | " << st.events << " |\n";
  for (int t = 1; t <= 7; ++t) {
    const auto n = st.by_type[static_cast<std::size_t>(t)];
    if (n > 0) {
      std::cout << "| &nbsp;&nbsp;" << to_string(static_cast<Event>(t)) << " | " << n << " |\n";
    }
  }
  std::cout << "| Level comparisons (price and size) | " << st.comparisons << " |\n"
            << "| **Events with any visible level wrong** | **" << st.mismatches << "** |\n"
            << "| **Events that could not be applied** | **" << st.errors << "** |\n"
            << "| Levels seeded from the first snapshot | " << st.seeded_levels << " |\n"
            << "| Levels rebuilt as they scrolled into view | " << st.revealed_levels << " |\n"
            << "| &nbsp;&nbsp;of which the replayed size was already right | "
            << st.revealed_already_right << " (" << pct(st.revealed_already_right, st.revealed_levels)
            << ") |\n"
            << "| Events on orders never seen (drawn from anonymous liquidity) | "
            << st.unknown_id_events << " |\n"
            << "| Book events emitted during the replay | " << main_run.digest_events << " |\n"
            << "| Event-stream digest (FNV-1a 64) | `0x" << std::hex << std::setw(16)
            << std::setfill('0') << main_run.digest << std::dec << std::setfill(' ') << "` |\n\n";
  if (st.first_mismatch != 0) {
    std::cout << "First mismatch at event " << st.first_mismatch << ": " << st.first_mismatch_detail
              << "\n\n";
  }
  if (st.first_error != 0) {
    std::cout << "First error at event " << st.first_error << ": " << st.first_error_detail << "\n\n";
  }

  const std::size_t execs = st.by_type[static_cast<std::size_t>(Event::ExecuteVisible)];
  std::cout << "## Visible executions through the matching engine\n\n"
            << "| Measure | Count | Share |\n|---|---:|---:|\n"
            << "| Named order was first in the queue: an IOC through the matcher filled exactly it | "
            << st.executions_matched << " | " << pct(st.executions_matched, execs) << " |\n"
            << "| Nasdaq filled an order that was not first in the queue | "
            << st.executions_out_of_order << " | " << pct(st.executions_out_of_order, execs)
            << " |\n\n";

  // Drop-one-message trials: an even mix of the four event types that change the book.
  std::cout << "## Drop-one-message check\n\n"
            << "| Dropped event | Type | First problem at | Caught at that event |\n"
            << "|---:|---|---:|:---:|\n";
  std::vector<std::vector<std::size_t>> by_type(8);
  for (std::size_t i = 0; i < main_run.types.size(); ++i) by_type[main_run.types[i]].push_back(i + 1);
  std::uint64_t seed = 20261120;
  std::size_t caught = 0;
  for (std::size_t t = 0; t < trials; ++t) {
    const auto& pool = by_type[1 + t % 4];  // submit, partial cancel, delete, visible execution
    if (pool.empty()) continue;
    const std::size_t k = pool[mix(seed) % pool.size()];
    const Run r = replay(msg_path, info, k, true, false);
    const std::size_t first = std::min(r.stats.first_mismatch == 0 ? SIZE_MAX : r.stats.first_mismatch,
                                       r.stats.first_error == 0 ? SIZE_MAX : r.stats.first_error);
    const bool hit = first == k;
    caught += hit ? 1 : 0;
    std::cout << "| " << k << " | " << to_string(static_cast<Event>(1 + t % 4)) << " | "
              << (first == SIZE_MAX ? std::string("never") : std::to_string(first)) << " | "
              << (hit ? "yes" : "**no**") << " |\n";
  }
  ok = ok && caught == trials;

  std::cout << "\n" << (ok ? "PASS" : "FAIL") << ": main replay " << std::fixed
            << std::setprecision(1) << main_run.seconds << " s (" << std::setprecision(0)
            << static_cast<double>(st.events) / main_run.seconds
            << " events/s including CSV parsing), " << caught << "/" << trials
            << " dropped messages caught at the dropped event\n";
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
} catch (const std::exception& e) {
  std::cerr << "lob_replay: " << e.what() << "\n";
  return 2;
}
