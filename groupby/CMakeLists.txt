set(GROUPBY_SUBDIR
    groupby
)

if(ENABLE_DPCPP)
    add_dpcpp_lib(groupby groupby.cpp)
    add_library_install(groupby)
    add_header_install(groupby.hpp ${GROUPBY_SUBDIR})

    add_dpcpp_lib(groupby_local groupby_local.cpp)
    add_library_install(groupby_local)
    add_header_install(groupby_local.hpp ${GROUPBY_SUBDIR})
endif()
