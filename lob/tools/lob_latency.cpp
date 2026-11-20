// Phase 2 step 4: engine-call latency on real Nasdaq order flow.
//
//   lob_latency data/AAPL_2012-06-21_34200000_57600000_message_10.csv [runs=5]
//
// Replays the LOBSTER day on both books (ordered-map levels, and tick-indexed array levels
// with one-cent slots) and times every order-book call the replay makes: adds (submissions),
// partial cancels, deletes and matches (an IOC that fills the executed order). The replay's
// own bookkeeping is not timed. Each book gets one untimed warm-up run, then `runs` timed
// runs, interleaved so machine noise lands on both books alike. Every percentile is reported
// as the median across runs with the [min, max] range beside it.
//
// Timings come from std::chrono::steady_clock around each call, so they include the cost of
// reading the clock; that overhead is measured and printed alongside.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lob/lobster.hpp"

using namespace lob;
using namespace lob::lobster;

namespace {

constexpr std::array<double, 4> kPcts{50.0, 90.0, 99.0, 99.9};
constexpr std::array<const char*, LatencyLog::kKinds> kKindNames{"add", "partial cancel",
                                                                 "delete", "match"};

struct RunStats {
  std::array<std::array<double, kPcts.size()>, LatencyLog::kKinds> pct{};
  std::array<std::size_t, LatencyLog::kKinds> count{};
  std::array<double, LatencyLog::kKinds> mean{};
  double seconds{0};  // whole replay, untimed run only
  std::size_t events{0};
};

double percentile(std::vector<std::uint32_t>& v, double p) {
  if (v.empty()) return 0;
  const auto k = static_cast<std::size_t>(p / 100.0 * static_cast<double>(v.size() - 1));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
  return v[k];
}

template <typename R>
RunStats run(const std::string& path, const FileInfo& info, std::int64_t tick, bool timed) {
  DayReader rd(path, info.orderbook_path, info.levels);
  Message m;
  Snapshot s;
  if (!rd.next(m, s)) throw std::runtime_error("empty file");
  R rp(info.levels, tick);
  LatencyLog log;
  for (auto& v : log.ns) v.reserve(250'000);
  if (timed) rp.set_latency_log(&log);
  rp.seed(s, m.ns);
  const auto t0 = std::chrono::steady_clock::now();
  while (rd.next(m, s)) rp.step(m, s);
  RunStats out;
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.events = rp.stats().events;
  if (rp.stats().mismatches != 0 || rp.stats().errors != 0) {
    throw std::runtime_error("replay did not match LOBSTER; run lob_replay for details");
  }
  for (std::size_t k = 0; k < LatencyLog::kKinds; ++k) {
    auto& v = log.ns[k];
    out.count[k] = v.size();
    double sum = 0;
    for (auto x : v) sum += x;
    out.mean[k] = v.empty() ? 0 : sum / static_cast<double>(v.size());
    for (std::size_t j = 0; j < kPcts.size(); ++j) out.pct[k][j] = percentile(v, kPcts[j]);
  }
  return out;
}

struct Summary {
  double median, lo, hi;
};
Summary summarize(std::vector<double> xs) {
  std::sort(xs.begin(), xs.end());
  return {xs[xs.size() / 2], xs.front(), xs.back()};
}
std::string fmt(const Summary& s, int prec = 0) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << s.median << " [" << s.lo << "–" << s.hi << "]";
  return os.str();
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 2) {
    std::cerr << "usage: lob_latency <TICKER_DATE_START_END_message_LEVELS.csv> [runs=5]\n";
    return 2;
  }
  const std::string path = argv[1];
  const int runs = argc > 2 ? std::stoi(argv[2]) : 5;
  const FileInfo info = describe_file(path);

  // Clock overhead: back-to-back reads, the floor under every timed call.
  std::vector<std::uint32_t> clock_ns(1'000'000);
  for (auto& x : clock_ns) {
    const auto a = std::chrono::steady_clock::now();
    const auto b = std::chrono::steady_clock::now();
    x = static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  }
  const double clock_p50 = percentile(clock_ns, 50), clock_p99 = percentile(clock_ns, 99);

  (void)run<Replay>(path, info, 1, true);  // warm-up: page faults, caches, allocator
  (void)run<ArrayReplay>(path, info, 100, true);
  std::vector<RunStats> map_runs, arr_runs, map_untimed, arr_untimed;
  for (int r = 0; r < runs; ++r) {
    map_runs.push_back(run<Replay>(path, info, 1, true));
    arr_runs.push_back(run<ArrayReplay>(path, info, 100, true));
    map_untimed.push_back(run<Replay>(path, info, 1, false));
    arr_untimed.push_back(run<ArrayReplay>(path, info, 100, false));
  }

  std::cout << "## Engine-call latency, " << info.ticker << " " << info.date
            << " (ns; median of " << runs << " runs [min–max])\n\n"
            << "| Call | Book | Calls | Mean | p50 | p90 | p99 | p99.9 |\n"
            << "|---|---|---:|---:|---:|---:|---:|---:|\n";
  for (std::size_t k = 0; k < LatencyLog::kKinds; ++k) {
    for (int b = 0; b < 2; ++b) {
      const auto& rs = b == 0 ? map_runs : arr_runs;
      std::vector<double> mean;
      for (const auto& r : rs) mean.push_back(r.mean[k]);
      std::cout << "| " << kKindNames[k] << " | " << (b == 0 ? "map" : "array") << " | "
                << rs[0].count[k] << " | " << fmt(summarize(mean), 1);
      for (std::size_t j = 0; j < kPcts.size(); ++j) {
        std::vector<double> xs;
        for (const auto& r : rs) xs.push_back(r.pct[k][j]);
        std::cout << " | " << fmt(summarize(xs));
      }
      std::cout << " |\n";
    }
  }

  std::cout << "\n## Whole-day replay, untimed (parsing, replay bookkeeping and comparison "
               "included)\n\n| Book | Seconds | Events per second |\n|---|---:|---:|\n";
  for (int b = 0; b < 2; ++b) {
    const auto& rs = b == 0 ? map_untimed : arr_untimed;
    std::vector<double> secs, eps;
    for (const auto& r : rs) {
      secs.push_back(r.seconds);
      eps.push_back(static_cast<double>(r.events) / r.seconds);
    }
    std::cout << "| " << (b == 0 ? "map" : "array") << " | " << fmt(summarize(secs), 3) << " | "
              << fmt(summarize(eps)) << " |\n";
  }
  std::cout << "\nClock overhead (two back-to-back steady_clock reads): p50 " << clock_p50
            << " ns, p99 " << clock_p99 << " ns. Every latency above includes it once.\n";
  return EXIT_SUCCESS;
} catch (const std::exception& e) {
  std::cerr << "lob_latency: " << e.what() << "\n";
  return 2;
}
