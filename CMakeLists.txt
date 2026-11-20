cmake_minimum_required(VERSION 3.25)
project(quant_stack VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QS_BUILD_TESTS "Build unit tests" ON)
option(QS_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(QS_WERROR "Treat compiler warnings as errors" OFF)
option(QS_BUILD_BENCHMARKS "Build Google Benchmark microbenchmarks (lob_bench)" OFF)
option(QS_BUILD_TOOLS "Build the command-line tools (lob_replay, lob_latency)" ON)
option(QS_BUILD_PYTHON "Build the pybind11 module (orderbook._lob); uv sync turns this on" OFF)

# Warnings and sanitizers go on one interface target so third-party code
# (GoogleTest) is not built with -Werror.
add_library(qs_options INTERFACE)
if(MSVC)
  target_compile_options(qs_options INTERFACE /W4 $<$<BOOL:${QS_WERROR}>:/WX>)
else()
  target_compile_options(qs_options INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    $<$<BOOL:${QS_WERROR}>:-Werror>)
  if(QS_SANITIZE)
    set(QS_SAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    target_compile_options(qs_options INTERFACE ${QS_SAN_FLAGS})
    target_link_options(qs_options INTERFACE ${QS_SAN_FLAGS})
  endif()
endif()

if(QS_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

if(QS_BUILD_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

add_subdirectory(lob)
