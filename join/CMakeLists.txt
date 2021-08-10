if(ENABLE_DPCPP)
    add_dpcpp_lib(slab_join slab_join.cpp)
    add_dpcpp_lib(join join.cpp)
    add_dpcpp_lib(nested_loop_join nested_join.cpp)
endif()


