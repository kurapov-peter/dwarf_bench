if(ENABLE_DPCPP)
    add_subdirectory(join_helpers)

    add_dpcpp_lib(slab_join slab_join.cpp)
    target_link_libraries(slab_join PRIVATE join_helpers_lib)

    add_dpcpp_lib(join join.cpp)
    target_link_libraries(join PRIVATE join_helpers_lib)

    add_dpcpp_lib(cuckoo_join cuckoo_join.cpp)
    target_link_libraries(cuckoo_join PRIVATE join_helpers_lib)
    
    add_dpcpp_lib(sort_join sort_join.cpp)
    target_link_libraries(sort_join PRIVATE join_helpers_lib)

    add_dpcpp_lib(nested_loop_join nested_join.cpp)
    target_link_libraries(nested_loop_join PRIVATE join_helpers_lib)
endif()

