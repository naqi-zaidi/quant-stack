add_executable(lob_tests
  test_types.cpp
  test_object_pool.cpp
  test_id_map.cpp
  test_order_book.cpp
  test_order_types.cpp
  test_cancel_modify.cpp
  test_event_stream.cpp
  test_properties.cpp
)
target_link_libraries(lob_tests PRIVATE lob::core qs_options GTest::gtest_main)
gtest_discover_tests(lob_tests)

# Step 2 gate: the property suite at scale, plus the planted-bug table for the README.
add_executable(lob_property_report property_report.cpp)
target_link_libraries(lob_property_report PRIVATE lob::core qs_options)
