cmake_minimum_required(VERSION 3.2)
project(dwarf_bench)

find_package(OpenCL REQUIRED)
find_package(oclhelpers 0.1.1 REQUIRED)

find_package(Boost 1.61 COMPONENTS program_options filesystem REQUIRED)

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
)

include(KernelFunctions)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(scan)
add_subdirectory(constant)

set(bench_libs 
  common
  scan
  constant
)

set(bench_sources bench.cpp register_dwarfs.cpp register_dwarfs.hpp)
add_executable(${PROJECT_NAME} ${bench_sources})
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::program_options ${bench_libs})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

add_kernel(vadd)
add_executable(simple simple.cpp)
target_link_libraries(simple PRIVATE common OpenCL::OpenCL oclhelpers::oclhelpers)
target_include_directories(simple PRIVATE ${PROJECT_SOURCE_DIR})
