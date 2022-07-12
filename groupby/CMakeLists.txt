if(ENABLE_DPCPP)
    set(groupby_sources
        groupby.cpp
    )
    set(groupby_global_sources
        groupby_global.cpp
        ${groupby_sources}
    )

    set(groupby_local_sources
        groupby_local.cpp
        ${groupby_sources}
    )

    set(groupby_perfect_sources
        perfect_groupby.cpp
        ${groupby_sources}
    )

    add_dpcpp_lib(groupby_global "${groupby_global_sources}")
    add_dpcpp_lib(groupby_local "${groupby_local_sources}")
    add_dpcpp_lib(groupby_perfect "${groupby_perfect_sources}")
endif()
