if(ENABLE_DPCPP)
    add_dpcpp_lib(groupby groupby.cpp)
    add_dpcpp_lib(groupby_local groupby_local.cpp)
endif()
