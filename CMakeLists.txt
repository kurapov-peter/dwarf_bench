cmake_minimum_required(VERSION 3.14)
set(dwarf_bench_version 0.1)
project(dwarf_bench VERSION ${dwarf_bench_version})

find_package(TBB REQUIRED)
find_package(CUDA)

option(ENABLE_DPCPP ON)
option(ENABLE_CUDA OFF)
option(ENABLE_EXAMPLES ON)

option(ENABLE_EXPERIMENTAL OFF)

if(ENABLE_EXPERIMENTAL)
  find_package(OpenCL REQUIRED)
  find_package(oclhelpers 0.1.3 REQUIRED)
endif()

if(ENABLE_DPCPP)
  find_package(oneDPL REQUIRED)
  add_definitions(-DDPCPP_ENABLED)
  if(ENABLE_EXPERIMENTAL)
    add_definitions(-DEXPERIMENTAL)
  endif()
endif()

if(CUDA_FOUND AND ENABLE_CUDA)
  message(STATUS "Using ${CUDA_VERSION} headers")
  if (${CUDA_VERSION_MAJOR} GREATER_EQUAL 11)
    add_definitions(-DCUDA_OCL_API_CHANGED)
  endif()
  add_definitions(-DCUDA_ENABLED)
endif()

find_package(Boost 1.61 COMPONENTS program_options filesystem REQUIRED)

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake
)

include(KernelFunctions)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(common)
add_subdirectory(scan)
add_subdirectory(constant)
add_subdirectory(sort)
add_subdirectory(reduce)
add_subdirectory(hash)
add_subdirectory(join)
add_subdirectory(probe)
add_subdirectory(groupby)

if(ENABLE_EXAMPLES)
  add_subdirectory(example)
endif()

set(bench_libs
  common
  tbbsort
)

if(ENABLE_EXPERIMENTAL)
  list(APPEND bench_libs constant)
endif()

if(ENABLE_DPCPP)
  add_subdirectory(common/dpcpp)

  list(APPEND bench_libs
    dpcpp_constant
    scan
    radix
    hash_build
    nested_loop_join
    join_helpers_lib
    join
    groupby
    groupby_local
    hash_build_non_bitmask
    omnisci_join
  )
  
  if(ENABLE_EXPERIMENTAL)
    list(APPEND bench_libs
      reduce
      standalone_scan
      slab_hash_build
      cuckoo_hash_build
      slab_join
      slab_probe
      omnisci_join
    )
  endif()
  if(ENABLE_CUDA)
  list(APPEND bench_libs
      dpcpp_constant_cuda
      scan_cuda
      radix_cuda
      omnisci_join_cuda
  )
  endif()
endif()

set(${CMAKE_CXX_FLAGS_DEBUG} "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(bench_sources main.cpp register_dwarfs.cpp register_dwarfs.hpp bench.cpp bench.hpp)
add_executable(dwarf_bench ${bench_sources})
target_link_libraries(dwarf_bench PRIVATE Boost::program_options ${bench_libs})
target_include_directories(dwarf_bench PRIVATE ${PROJECT_SOURCE_DIR})

if(ENABLE_EXPERIMENTAL)
  add_kernel(vadd)
  add_executable(simple simple.cpp)
  target_link_libraries(simple PRIVATE common OpenCL::OpenCL oclhelpers::oclhelpers)
  target_include_directories(simple PRIVATE ${PROJECT_SOURCE_DIR})
endif()

option(ENABLE_TESTS ON)
if(ENABLE_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.11.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_subdirectory(tests)
endif()

add_library(dbench ${bench_sources})
target_link_libraries(dbench PRIVATE ${bench_libs})
target_include_directories(dbench PRIVATE ${PROJECT_SOURCE_DIR})
set_target_properties(dbench PROPERTIES PUBLIC_HEADER "bench.hpp")

include(GNUInstallDirs)

install(TARGETS ${bench_libs} dbench dwarf_bench
  EXPORT dbench
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT dbench 
  FILE dbench.cmake
  NAMESPACE dbench::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dbench
)

# Just for convenience
export(EXPORT dbench 
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/dbench/dbench.cmake"
  NAMESPACE dbench::
)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/dbenchConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dbench
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/dbenchConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/dbenchConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dbench
)

set_property(TARGET dbench PROPERTY VERSION ${dwarf_bench_version})
set_property(TARGET dbench PROPERTY SOVERSION 0)
set_property(TARGET dbench PROPERTY
  INTERFACE_dbench_MAJOR_VERSION 0)
set_property(TARGET dbench APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING dbench_MAJOR_VERSION
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/dbenchConfigVersion.cmake"
  VERSION "${dwarf_bench_version}"
  COMPATIBILITY AnyNewerVersion
)
