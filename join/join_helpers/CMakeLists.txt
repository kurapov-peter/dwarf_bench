set(join_helpers_sources

    join_helpers.hpp
)

set(JOIN_HELPERS_LIBS join_helpers_lib)

add_library(${JOIN_HELPERS_LIBS} ${join_helpers_sources})
target_link_libraries(${JOIN_HELPERS_LIBS} 
    PRIVATE common
)