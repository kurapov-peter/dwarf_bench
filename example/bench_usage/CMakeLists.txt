cmake_minimum_required(VERSION 3.14)
project(dwarf_bench_test)

find_package(TBB REQUIRED)
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
find_package(OpenCL REQUIRED)
find_package(oclhelpers 0.1.3 REQUIRED)
find_package(oneDPL REQUIRED)
find_package(dbench REQUIRED)

add_executable(bench_usage main.cpp)
target_link_libraries(bench_usage dbench::dbench intlc)
