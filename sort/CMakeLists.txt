add_tbb_lib(tbbsort tbbsort.cpp)

if(ENABLE_DPCPP)
    add_dpcpp_lib(radix radix.cpp)
    if(ENABLE_CUDA)
        add_dpcpp_cuda_lib(radix radix_cuda.cpp)
    endif()
endif()