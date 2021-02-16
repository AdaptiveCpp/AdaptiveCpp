if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
endif()
add_subdirectory(runtime)

set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME ${HIPSYCL_RT_LIBRARY_OUTPUT_NAME} PARENT_SCOPE)
