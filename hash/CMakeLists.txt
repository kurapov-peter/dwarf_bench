if(ENABLE_DPCPP)
    add_dpcpp_lib(hash_build hash_build.cpp)
    add_dpcpp_lib(hash_build_non_bitmask hash_build_non_bitmask.cpp)
    if(ENABLE_EXPERIMENTAL)
        add_dpcpp_lib(slab_hash_build slab_hash_build.cpp)
        add_dpcpp_lib(cuckoo_hash_build cuckoo_hash_build.cpp)
    endif()
endif()
