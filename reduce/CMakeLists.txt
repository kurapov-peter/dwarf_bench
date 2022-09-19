set(REDUCE_SUBDIR
    reduce
)

if(ENABLE_DPCPP)
    add_dpcpp_lib(reduce reduce.cpp)
    if(ENABLE_CUDA)
        add_dpcpp_cuda_lib(reduce reduce.cpp)
        add_library_install(reduce_cuda)
    endif()

    add_library_install(reduce)
    add_header_install(reduce.hpp ${REDUCE_SUBDIR})
endif()