enable_testing()

add_executable(scan_tests scan_tests.cpp)
target_link_libraries(scan_tests gtest common scan)

add_test(scan_tests scan_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)