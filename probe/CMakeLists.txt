if(ENABLE_DPCPP)
    add_dpcpp_lib(slab_probe slab_probe.cpp)
    add_dpcpp_lib(cuckoo_probe cuckoo_probe.cpp)
endif()