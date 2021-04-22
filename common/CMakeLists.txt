set(common_sources
    oclhelpers.cpp
    registry.cpp
    result.cpp
    common.cpp
    meter.cpp

    common.hpp
    meter.hpp
    dwarf.hpp
    oclhelpers.hpp
    registry.hpp
    result.hpp
)

set(COMMON_LIB common)

add_library(${COMMON_LIB} ${common_sources})
target_link_libraries(${COMMON_LIB} PUBLIC OpenCL::OpenCL PRIVATE Boost::filesystem ${CMAKE_DL_LIBS})
