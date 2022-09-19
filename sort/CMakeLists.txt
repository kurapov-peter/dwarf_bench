set(SORT_SUBDIR
    sort
)
add_tbb_lib(tbbsort tbbsort.cpp)
add_library_install(tbbsort)
add_header_install(tbbsort.hpp ${SORT_SUBDIR})

if(ENABLE_DPCPP)
    add_dpcpp_lib(radix radix.cpp)
    add_library_install(radix)
    add_header_install(radix.hpp ${SORT_SUBDIR})

    if(ENABLE_CUDA)
        add_dpcpp_cuda_lib(radix radix_cuda.cpp)
        add_library_install(radix_cuda)
    endif()
endif()
