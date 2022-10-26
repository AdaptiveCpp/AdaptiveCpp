set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(common)

list( APPEND CMAKE_INSTALL_RPATH
  ${CMAKE_INSTALL_PREFIX}/lib
  ${CMAKE_INSTALL_PREFIX}/lib/hipSYCL/llvm-to-backend)

if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
  add_subdirectory(libkernel)
endif()

add_subdirectory(runtime)
add_subdirectory(tools)

set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME ${HIPSYCL_RT_LIBRARY_OUTPUT_NAME} PARENT_SCOPE)
set(hipSYCL_OpenMP_CXX_LIBRARIES ${hipSYCL_OpenMP_CXX_LIBRARIES} PARENT_SCOPE)
