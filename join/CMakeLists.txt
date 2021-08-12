if(ENABLE_DPCPP)
    add_dpcpp_lib(join join.cpp)
    add_dpcpp_lib(cuckoo_join cuckoo_join.cpp)
endif()