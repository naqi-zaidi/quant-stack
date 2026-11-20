add_library(lob_core STATIC src/order_book.cpp src/lobster.cpp)
add_library(lob::core ALIAS lob_core)
target_include_directories(lob_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lob_core PUBLIC cxx_std_20)
target_link_libraries(lob_core PRIVATE qs_options)
# Position-independent so the same archive can be linked into the pybind11 module (step 5).
set_target_properties(lob_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(QS_BUILD_TOOLS)
  # Step 3 gate: LOBSTER replay (needs a LOBSTER sample in data/, see data/README.md).
  add_executable(lob_replay tools/lob_replay.cpp)
  target_link_libraries(lob_replay PRIVATE lob::core qs_options)

  # Step 4: engine-call latency percentiles on the same real day, map vs array levels.
  add_executable(lob_latency tools/lob_latency.cpp)
  target_link_libraries(lob_latency PRIVATE lob::core qs_options)
endif()

# Step 5: the Python module. Built by `uv sync` through scikit-build-core (see pyproject.toml).
if(QS_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(QS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# Step 4: Google Benchmark microbenchmarks (cmake -DQS_BUILD_BENCHMARKS=ON; on in `release`).
if(QS_BUILD_BENCHMARKS)
  add_executable(lob_bench bench/bench_book.cpp)
  target_link_libraries(lob_bench PRIVATE lob::core qs_options benchmark::benchmark)
endif()
