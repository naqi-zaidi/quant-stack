// Phase 2 step 2 gate. Runs the property suite at scale and prints the README tables:
//   1. the engine against every property, and against the reference model, per regime;
//   2. each planted bug: which property caught it, how fast, and its shrunk counterexample.
// Exits non-zero if the engine breaks any property or any planted bug survives.
//
//   lob_property_report [flows_per_regime=40] [calls_per_flow=2000]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "property_harness.hpp"

using namespace lob::testing_helpers;

int main(int argc, char** argv) {
  const std::uint64_t flows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 40;
  const std::size_t calls = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2000;
  const auto t0 = std::chrono::steady_clock::now();
  bool ok = true;

  std::cout << "## Engine vs every property, and vs the reference model\n\n"
            << "| Regime | Flows | Calls | Accepted | Trades | Property violations | "
               "Mismatches vs model |\n"
            << "|---|---:|---:|---:|---:|---:|---:|\n";
  std::size_t t_calls = 0, t_acc = 0, t_trades = 0, t_viol = 0, t_diff = 0;
  for (const Regime& rg : kRegimes) {
    std::size_t n_calls = 0, n_acc = 0, n_trades = 0, n_viol = 0, n_diff = 0;
    for (std::uint64_t seed = 1; seed <= flows; ++seed) {
      const auto ops = generate_flow(rg, seed, calls);
      EngineUnderTest eng;
      const FlowResult fr = run_properties(ops, eng);
      n_calls += fr.calls;
      n_acc += fr.accepted;
      n_trades += fr.trades;
      if (fr.violation) {
        ++n_viol;
        std::cerr << rg.name << " seed " << seed << " call " << fr.step + 1 << ": "
                  << fr.violation->property << ": " << fr.violation->detail << "\n";
      }
      if (run_differential(ops).violation) ++n_diff;
    }
    std::cout << "| " << rg.name << " | " << flows << " | " << n_calls << " | " << n_acc
              << " | " << n_trades << " | " << n_viol << " | " << n_diff << " |\n";
    t_calls += n_calls;
    t_acc += n_acc;
    t_trades += n_trades;
    t_viol += n_viol;
    t_diff += n_diff;
  }
  std::cout << "| **all** | " << flows * kRegimes.size() << " | " << t_calls << " | " << t_acc
            << " | " << t_trades << " | " << t_viol << " | " << t_diff << " |\n\n";
  ok = ok && t_viol == 0 && t_diff == 0;

  std::cout << "## Planted bugs\n\n"
            << "| Bug | Caught by | Calls until caught | Shrunk to | Model diff at call |\n"
            << "|---|---|---:|---:|---:|\n";
  std::string examples;
  for (const SeededBug& bug : kSeededBugs) {
    const Hunt h = hunt(bug.m, 20, calls);
    if (!h.caught) {
      ok = false;
      std::cout << "| " << bug.name << " | **survived** " << h.flows << " flows | | | |\n";
      continue;
    }
    ok = ok && h.differential_at > 0;
    std::cout << "| " << bug.name << " | " << h.violation.property << " | " << h.calls_to_catch
              << " | " << h.shrunk.size() << (h.shrunk.size() == 1 ? " call" : " calls") << " | "
              << h.differential_at << " |\n";
    if (bug.m == Mutation::LifoQueue || bug.m == Mutation::SizeDownLosesPriority) {
      examples += "\n" + std::string(bug.name) + ":\n" + describe(h.shrunk, bug.m);
    }
  }
  std::cout << "\n## Example shrunk counterexamples\n" << examples;

  const double secs =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "\n" << (ok ? "PASS" : "FAIL") << " in " << secs << " s\n";
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
