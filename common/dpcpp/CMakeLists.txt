set(DPCPP_COMMON_SUBDIR
    "${COMMON_SUBDIR}/dpcpp"
)

set(dpcpp_common_headers
    dpcpp_common.hpp
    hashtable.hpp
    cuckoo_hashtable.hpp
    slab_hash.hpp
    hashfunctions.hpp

    omnisci_hashtable.hpp
)

set(dpcpp_common_sources
    dpcpp_common.cpp
    ${dpcpp_common_headers}
)

add_library(dpcpp_common ${dpcpp_common_sources})
target_include_directories(dpcpp_common PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(dpcpp_common 
    PRIVATE sycl
)
target_compile_options(dpcpp_common PRIVATE -fsycl)
target_link_options(dpcpp_common PRIVATE -fsycl)
add_library_install(dpcpp_common)
add_header_install("${dpcpp_common_headers}" "${DPCPP_COMMON_SUBDIR}")
