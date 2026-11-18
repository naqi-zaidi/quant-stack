add_executable(lob_tests test_types.cpp)
target_link_libraries(lob_tests PRIVATE lob::core qs_options GTest::gtest_main)
gtest_discover_tests(lob_tests)
