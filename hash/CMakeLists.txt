set(HASH_SUBDIR
    hash
)

if(ENABLE_DPCPP)
    add_dpcpp_lib(hash_build hash_build.cpp)
    add_library_install(hash_build)
    add_header_install(hash_build.hpp ${HASH_SUBDIR})

    add_dpcpp_lib(hash_build_non_bitmask hash_build_non_bitmask.cpp)
    add_library_install(hash_build_non_bitmask)
    add_header_install(hash_build_non_bitmask.hpp ${HASH_SUBDIR})

    add_dpcpp_lib(cuckoo_hash_build cuckoo_hash_build.cpp)
    add_library_install(cuckoo_hash_build)
    add_header_install(cuckoo_hash_build.hpp ${HASH_SUBDIR})

    if(ENABLE_EXPERIMENTAL)
        add_dpcpp_lib(slab_hash_build slab_hash_build.cpp)
        add_library_install(slab_hash_build)
        add_header_install(slab_hash_build.hpp ${HASH_SUBDIR})
    endif()
endif()
