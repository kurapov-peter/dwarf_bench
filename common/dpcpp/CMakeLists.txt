set(dpcpp_common_sources
    dpcpp_common.cpp

    dpcpp_common.hpp
    hashtable.hpp
    slab_hash.hpp
)

add_library(dpcpp_common ${dpcpp_common_sources})
target_include_directories(dpcpp_common PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(dpcpp_common 
    PRIVATE sycl
)
target_compile_options(dpcpp_common PRIVATE -fsycl)
target_link_options(dpcpp_common PRIVATE -fsycl)

