add_kernel(scan)

set(scan_sources
    scan.cpp
    scan.hpp
)

add_library(scan ${scan_sources})
target_link_libraries(scan PUBLIC common OpenCL::OpenCL oclhelpers::oclhelpers)
target_include_directories(scan PRIVATE ${PROJECT_SOURCE_DIR})

if(ENABLE_DPCPP)
    set(dpl_scan_sources
        dplscan.cpp
        scan.hpp
    )

    add_library(dplscan SHARED ${dpl_scan_sources})
    target_link_libraries(dplscan common dpcpp_common oneDPL)
    target_include_directories(dplscan PRIVATE ${PROJECT_SOURCE_DIR})

    set(dpl_scan_cuda_sources
        dplscan_cuda.cpp
        scan.hpp
    )
    add_library(dplscan_cuda SHARED ${dpl_scan_cuda_sources})
    target_link_libraries(dplscan_cuda common dpcpp_common oneDPL)
    target_include_directories(dplscan_cuda PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(dplscan_cuda PRIVATE -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
    target_link_options(dplscan_cuda PRIVATE -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)

endif()