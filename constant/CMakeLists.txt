add_kernel(constant)

set(sources
    constant.cpp
    constant_capi.cpp
    constant.hpp
)

add_library(constant ${sources})
target_link_libraries(constant PRIVATE common OpenCL::OpenCL)
target_include_directories(constant PRIVATE ${PROJECT_SOURCE_DIR})