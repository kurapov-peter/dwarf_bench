add_kernel(scan)

set(scan_sources
    scan.cpp
    scan.hpp
)
add_library(standalone_scan ${scan_sources})
target_link_libraries(standalone_scan PRIVATE common OpenCL::OpenCL oclhelpers::oclhelpers)
target_include_directories(standalone_scan PRIVATE ${PROJECT_SOURCE_DIR})

if(ENABLE_DPCPP)
    add_dpcpp_lib(scan dplscan.cpp)
    if(ENABLE_CUDA)
        add_dpcpp_cuda_lib(scan dplscan_cuda.cpp)
    endif()
endif()