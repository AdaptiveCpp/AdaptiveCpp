set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE AND NOT WIN32)
  # symbolic functions causes references in our libraries to be resolved
  # to definitions within the libraries themselves, if possible.
  # If both runtime library and SYCL headers define some of the same
  # symbols, this can prevent code using hijacked USM malloc for standard
  # parallelism to bleed into code that is executed within the runtime, where
  # USM is a big no-no.
  set(HIPSYCL_STDPAR_RT_LINKER_FLAGS "-Wl,-Bsymbolic-functions")
  set(base $ORIGIN)
else()
  set(HIPSYCL_STDPAR_RT_LINKER_FLAGS "")
  set(base @loader_path)
endif()

add_subdirectory(common)

set(WITH_LLVM_TO_AMDGPU_AMDHSA false)
set(WITH_LLVM_TO_PTX false)
set(WITH_LLVM_TO_SPIRV false)
set(WITH_LLVM_TO_HOST false)

if(WITH_SSCP_COMPILER)
  if(WITH_LEVEL_ZERO_BACKEND OR WITH_OPENCL_BACKEND)
    set(WITH_LLVM_TO_SPIRV true)
  endif()

  if(WITH_CUDA_BACKEND)
    set(WITH_LLVM_TO_PTX true)
  endif()

  if(WITH_ROCM_BACKEND)
    set(WITH_LLVM_TO_AMDGPU_AMDHSA true)
  endif()

  set(WITH_LLVM_TO_HOST true)
endif()

if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
  add_subdirectory(libkernel)
endif()

set(ACPP_RT_SANITIZE "" CACHE STRING
    "Enable building AdaptiveCpp runtime with specified sanitizers")

if(NOT ${ACPP_RT_SANITIZE} STREQUAL "")
  set(ACPP_RT_SANITIZE_FLAGS "-fsanitize=${ACPP_RT_SANITIZE}")
else()
  set(ACPP_RT_SANITIZE_FLAGS "")
endif()


add_subdirectory(runtime)
add_subdirectory(tools)

set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME ${HIPSYCL_RT_LIBRARY_OUTPUT_NAME} PARENT_SCOPE)
set(hipSYCL_OpenMP_CXX_LIBRARIES ${hipSYCL_OpenMP_CXX_LIBRARIES} PARENT_SCOPE)
