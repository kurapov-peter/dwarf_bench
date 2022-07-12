if(ENABLE_DPCPP)
    set(groupby_sources
        groupby.cpp
    )
    set(groupby_global
        groupby_global.cpp
        ${groupby_sources}
    )

    set(groupby_local
        groupby_local.cpp
    )

    add_dpcpp_lib(groupby_global "${groupby_global}")
    add_dpcpp_lib(groupby_local "${groupby_local}")
endif()
