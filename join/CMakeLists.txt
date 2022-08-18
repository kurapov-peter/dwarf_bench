if(ENABLE_DPCPP)
    add_subdirectory(join_helpers)

    add_dpcpp_lib(slab_join slab_join.cpp)
    target_link_libraries(slab_join PRIVATE join_helpers_lib)

    add_dpcpp_lib(join join.cpp)
    target_link_libraries(join PRIVATE join_helpers_lib)

    add_dpcpp_lib(nested_loop_join nested_join.cpp)
    target_link_libraries(nested_loop_join PRIVATE join_helpers_lib)

    if(ENABLE_EXPERIMENTAL)
        add_dpcpp_lib(omnisci_join join_omnisci.cpp)
        target_link_libraries(omnisci_join PRIVATE join_helpers_lib)
    endif()
endif()

