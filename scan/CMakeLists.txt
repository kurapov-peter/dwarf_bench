add_kernel(scan)

set(SCAN_SUBDIR
    scan
)
set(scan_sources
    scan.cpp
    scan.hpp
)
add_library(standalone_scan ${scan_sources})
target_link_libraries(standalone_scan PRIVATE common OpenCL::OpenCL oclhelpers::oclhelpers)
target_include_directories(standalone_scan PRIVATE ${PROJECT_SOURCE_DIR})
add_library_install(standalone_scan)

if(ENABLE_DPCPP)
    add_dpcpp_lib(scan dplscan.cpp)
    add_library_install(scan)
    if(ENABLE_CUDA)
        add_dpcpp_cuda_lib(scan dplscan_cuda.cpp)
        add_library_install(scan_cuda)
    endif()
endif()

add_header_install(scan.hpp ${SCAN_SUBDIR})
install(FILES scan.cl DESTINATION kernel)
