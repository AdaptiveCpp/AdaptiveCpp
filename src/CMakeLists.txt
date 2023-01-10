set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(common)

list( APPEND CMAKE_INSTALL_RPATH
  ${CMAKE_INSTALL_PREFIX}/lib
  ${CMAKE_INSTALL_PREFIX}/lib/hipSYCL/llvm-to-backend)

set(WITH_LLVM_TO_AMDGPU_AMDHSA false)
set(WITH_LLVM_TO_PTX false)
set(WITH_LLVM_TO_SPIRV false)

if(WITH_SSCP_COMPILER)
  if(WITH_LEVEL_ZERO_BACKEND)
    set(WITH_LLVM_TO_SPIRV true)
  endif()

  if(WITH_CUDA_BACKEND)
    set(WITH_LLVM_TO_PTX true)
  endif()

  if(WITH_ROCM_BACKEND)
    set(WITH_LLVM_TO_AMDGPU_AMDHSA true)
  endif()
endif()

if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
  add_subdirectory(libkernel)
endif()

add_subdirectory(runtime)
add_subdirectory(tools)

set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME ${HIPSYCL_RT_LIBRARY_OUTPUT_NAME} PARENT_SCOPE)
set(hipSYCL_OpenMP_CXX_LIBRARIES ${hipSYCL_OpenMP_CXX_LIBRARIES} PARENT_SCOPE)
