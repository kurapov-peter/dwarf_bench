if(ENABLE_DPCPP)
    add_dpcpp_lib(hash_build hash_build.cpp)
    add_dpcpp_lib(slab_hash_build slab_hash_build.cpp)
endif()