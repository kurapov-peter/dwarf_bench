add_kernel(constant)

set(sources
    constant.cpp
    constant_capi.cpp
    constant.hpp
)

add_library(constant ${sources})
target_link_libraries(constant PRIVATE common OpenCL::OpenCL oclhelpers::oclhelpers)
target_include_directories(constant PRIVATE ${PROJECT_SOURCE_DIR})

if(ENABLE_DPCPP)
    set(dpcpp_sources
        constant_dpcpp.cpp
        constant.hpp
    )
    set(dpcpp_common_libs common dpcpp_common sycl)
    if(EXPLICIT_DPL)
        list(APPEND dpcpp_common_libs oneDPL)
    endif()
    add_library(dpcpp_constant SHARED ${dpcpp_sources})
    target_link_libraries(dpcpp_constant PRIVATE ${dpcpp_common_libs})
    target_include_directories(dpcpp_constant PRIVATE ${PROJECT_SOURCE_DIR})
    target_compile_options(dpcpp_constant PRIVATE -fsycl)
    target_link_options(dpcpp_constant PRIVATE -fsycl)

    if(ENABLE_CUDA)
        set(dpcpp_cuda_sources
            constant_dpcpp_cuda.cpp
            constant.hpp
        )
        add_library(dpcpp_constant_cuda SHARED ${dpcpp_cuda_sources})
        target_link_libraries(dpcpp_constant_cuda PRIVATE ${common_dpcpp_libs})
        target_include_directories(dpcpp_constant_cuda PRIVATE ${PROJECT_SOURCE_DIR})
        target_compile_options(dpcpp_constant_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
        target_link_options(dpcpp_constant_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
    endif()
endif()