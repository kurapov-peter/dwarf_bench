function(add_kernel name)
  configure_file(${name}.cl ${CMAKE_CURRENT_BINARY_DIR}/${name}.cl COPYONLY)
endfunction()