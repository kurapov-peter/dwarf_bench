if(ENABLE_DPCPP)
    add_dpcpp_lib(radix radix.cpp)
    add_dpcpp_cuda_lib(radix radix_cuda.cpp)
endif()