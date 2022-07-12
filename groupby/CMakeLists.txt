if(ENABLE_DPCPP)
    add_dpcpp_lib(groupby_global groupby_global.cpp)
    add_dpcpp_lib(groupby_local groupby_local.cpp)
endif()
