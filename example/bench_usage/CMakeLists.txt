cmake_minimum_required(VERSION 3.14)
project(dwarf_bench_test)

find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(dbench REQUIRED)

add_executable(bench_usage main.cpp)
target_link_libraries(bench_usage dbench::dbench)
