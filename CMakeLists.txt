cmake_minimum_required(VERSION 3.2)
project(dwarf_bench)

find_package(OpenCL REQUIRED)
message(STATUS "OpenCL includes: ${OpenCL_INCLUDE_DIR}")

function(add_kernel name)
  configure_file(${name}.cl ${CMAKE_CURRENT_BINARY_DIR}/${name}.cl COPYONLY)
endfunction()

#configure_file(vadd.cl ${CMAKE_CURRENT_BINARY_DIR}/vadd.cl COPYONLY)
#configure_file(vadd.cl ${CMAKE_CURRENT_BINARY_DIR}/vadd.cl COPYONLY)

add_kernel(vadd)
add_kernel(scan)

add_executable(simple simple.cpp)
target_link_libraries(simple OpenCL::OpenCL)

add_executable(scan scan.cpp)
target_link_libraries(scan OpenCL::OpenCL)
