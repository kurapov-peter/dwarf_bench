if(ENABLE_DPCPP)
    set(JOIN_SUBDIR join)
    add_subdirectory(join_helpers)

    add_dpcpp_lib(join join.cpp)
    target_link_libraries(join PRIVATE join_helpers_lib)
    add_library_install(join)
    add_header_install(join.hpp ${JOIN_SUBDIR})

    add_dpcpp_lib(nested_loop_join nested_join.cpp)
    target_link_libraries(nested_loop_join PRIVATE join_helpers_lib)
    add_library_install(nested_loop_join)
    add_header_install(nested_join.hpp ${JOIN_SUBDIR})

    if(ENABLE_EXPERIMENTAL)
        add_dpcpp_lib(omnisci_join join_omnisci.cpp)
        target_link_libraries(omnisci_join PRIVATE join_helpers_lib)
        add_library_install(omnisci_join)
        add_header_install(omnisci_join.hpp ${JOIN_SUBDIR})

        add_dpcpp_lib(slab_join slab_join.cpp)
        target_link_libraries(slab_join PRIVATE join_helpers_lib)
        add_library_install(slab_join)
        add_header_install(slab_join.hpp ${JOIN_SUBDIR})
    endif()
endif()

