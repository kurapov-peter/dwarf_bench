enable_testing()

add_executable(dwarf_tests dwarf_tests.cpp utils.cpp)

target_link_libraries(dwarf_tests gtest ${bench_libs} oclhelpers::oclhelpers)

target_include_directories(dwarf_tests PRIVATE ${PROJECT_SOURCE_DIR})

target_compile_options(dwarf_tests PRIVATE -fsycl)
target_link_options(dwarf_tests PRIVATE -fsycl)

add_test(dwarf_tests dwarf_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests/dwarf_tests)