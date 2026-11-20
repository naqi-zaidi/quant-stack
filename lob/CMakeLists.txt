add_library(lob_core STATIC src/order_book.cpp src/lobster.cpp)
add_library(lob::core ALIAS lob_core)
target_include_directories(lob_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(lob_core PUBLIC cxx_std_20)
target_link_libraries(lob_core PRIVATE qs_options)
# Position-independent so the same archive can be linked into the pybind11 module (step 5).
set_target_properties(lob_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Step 3 gate: LOBSTER replay (needs a LOBSTER sample in data/, see data/README.md).
add_executable(lob_replay tools/lob_replay.cpp)
target_link_libraries(lob_replay PRIVATE lob::core qs_options)

if(QS_BUILD_TESTS)
  add_subdirectory(tests)
endif()
