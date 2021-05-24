set(dpcpp_common_sources
    dpcpp_common.cpp

    dpcpp_common.hpp
)

add_library(dpcpp_common ${dpcpp_common_sources})
target_include_directories(dpcpp_common PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(dpcpp_common 
    PUBLIC sycl
)
target_compile_options(dpcpp_common PUBLIC -fsycl)
target_link_options(dpcpp_common PUBLIC -fsycl)

