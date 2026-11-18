# Header-only for now; becomes a compiled library when the matching engine lands (Phase 2).
add_library(lob_core INTERFACE)
add_library(lob::core ALIAS lob_core)
target_include_directories(lob_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lob_core INTERFACE cxx_std_20)

if(QS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
