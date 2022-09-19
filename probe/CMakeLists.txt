set(PROBE_SUBDIR
    probe
)

if(ENABLE_DPCPP)
    if(ENABLE_EXPERIMENTAL)
        add_dpcpp_lib(slab_probe slab_probe.cpp)
        add_library_install(slab_probe)
        add_header_install(slab_probe.hpp ${PROBE_SUBDIR})
    endif()
endif()
