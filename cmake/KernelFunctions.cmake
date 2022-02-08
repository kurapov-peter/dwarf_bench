function(add_kernel name)
  configure_file(${name}.cl ${CMAKE_CURRENT_BINARY_DIR}/${name}.cl COPYONLY)
endfunction()

set(common_dpcpp_libs common dpcpp_common sycl oclhelpers::oclhelpers)
set(common_tbb_libs common tbb)

if(EXPLICIT_DPL)
  list(APPEND common_dpcpp_libs oneDPL)
endif()

function(add_dpcpp_lib name sources)
  message(STATUS "Adding dpcpp library ${name}")
  add_library(${name} SHARED ${sources})
  target_link_libraries(${name} PRIVATE ${common_dpcpp_libs})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(${name} PRIVATE -fsycl)
  target_link_options(${name} PRIVATE -fsycl)
endfunction()

function(add_tbb_lib name sources)
  message(STATUS "Adding tbb library ${name}")
  add_library(${name} SHARED ${sources})
  target_link_libraries(${name} PRIVATE ${common_tbb_libs})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
endfunction()

function(add_dpcpp_cuda_lib name sources)
  add_library(${name}_cuda SHARED ${sources})
  target_link_libraries(${name}_cuda PRIVATE ${common_dpcpp_libs})
  target_include_directories(${name}_cuda PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(${name}_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
  target_link_options(${name}_cuda PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice)
endfunction()

function(add_example name sources)
  message(STATUS "Adding example ${name}")
  add_executable(${name} ${sources})
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR})
  target_compile_options(${name} PRIVATE -fsycl)
  target_link_options(${name} PRIVATE -fsycl)
endfunction()