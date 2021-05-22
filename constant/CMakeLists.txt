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
    add_library(dpcpp_constant ${dpcpp_sources})
    target_link_libraries(dpcpp_constant common sycl)
    target_include_directories(dpcpp_constant PRIVATE ${PROJECT_SOURCE_DIR})
    #target_include_directories(dpcpp_constant PRIVATE /data/dpcpp/llvm/build/include/sycl)
    target_compile_options(dpcpp_constant PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
    #target_compile_options(dpcpp_constant PUBLIC -fsycl)
    target_link_options(dpcpp_constant PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
endif()