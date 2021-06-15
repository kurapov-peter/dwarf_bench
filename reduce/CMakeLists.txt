if(ENABLE_DPCPP)
    add_dpcpp_lib(reduce reduce.cpp)
    #add_dpcpp_cuda_lib(reduce reduce.cpp)
endif()