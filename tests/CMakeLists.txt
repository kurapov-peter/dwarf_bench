enable_testing()

add_executable(scan_tests scan_tests.cpp)
target_link_libraries(scan_tests gtest scan)
target_include_directories(scan_tests PRIVATE ${PROJECT_SOURCE_DIR})

add_test(scan_tests scan_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)