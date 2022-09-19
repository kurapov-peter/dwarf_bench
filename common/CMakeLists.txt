set(common_headers
    common.hpp
    meter.hpp
    dwarf.hpp
    registry.hpp
    result.hpp
    options.hpp
)

set(common_sources
    registry.cpp
    result.cpp
    common.cpp
    meter.cpp
    options.cpp

    ${common_headers}
)
set(COMMON_SUBDIR common)
set(COMMON_LIB common)

add_library(${COMMON_LIB} ${common_sources})
target_link_libraries(${COMMON_LIB} 
    #PUBLIC OpenCL::OpenCL oclhelpers::oclhelpers
    PRIVATE Boost::filesystem ${CMAKE_DL_LIBS}
)
add_library_install(${COMMON_LIB})
add_header_install("${common_headers}" "${COMMON_SUBDIR}")
