add_kernel(scan)

set(scan_sources
    scan.cpp
    scan.hpp
)

add_library(scan ${scan_sources})
target_link_libraries(scan PRIVATE common OpenCL::OpenCL)
target_include_directories(scan PRIVATE ${PROJECT_SOURCE_DIR})