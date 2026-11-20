add_executable(lob_tests
  test_types.cpp
  test_object_pool.cpp
  test_id_map.cpp
  test_order_book.cpp
  test_order_types.cpp
  test_cancel_modify.cpp
  test_event_stream.cpp
  test_properties.cpp
  test_lobster.cpp
  test_golden.cpp
  test_digest.cpp
)
target_link_libraries(lob_tests PRIVATE lob::core qs_options GTest::gtest_main)
# Snapshot tests read and (with UPDATE_GOLDEN=1) write lob/tests/golden/*.txt.
target_compile_definitions(lob_tests PRIVATE LOB_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
gtest_discover_tests(lob_tests)

# Step 2 gate: the property suite at scale, plus the planted-bug table for the README.
add_executable(lob_property_report property_report.cpp)
target_link_libraries(lob_property_report PRIVATE lob::core qs_options)
