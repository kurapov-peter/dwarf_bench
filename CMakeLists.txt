cmake_minimum_required(VERSION 3.2)
project(dwarf_bench)

find_package(OpenCL REQUIRED)

configure_file(vadd.cl ${CMAKE_CURRENT_BINARY_DIR}/vadd.cl COPYONLY)

add_executable(simple simple.cpp)
target_link_libraries(simple OpenCL::OpenCL)
