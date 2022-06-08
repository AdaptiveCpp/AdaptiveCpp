if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
endif()
add_subdirectory(runtime)
add_subdirectory(tools)

set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME ${HIPSYCL_RT_LIBRARY_OUTPUT_NAME} PARENT_SCOPE)
set(hipSYCL_OpenMP_CXX_LIBRARIES ${hipSYCL_OpenMP_CXX_LIBRARIES} PARENT_SCOPE)
