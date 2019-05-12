cmake_minimum_required (VERSION 3.5)
project(hipSYCL)


set(HIPSYCL_DEVICE_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc)
set(HIPSYCL_SOURCE_DIR ${PROJECT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT HIPSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(HIPSYCL_DEBUG_LEVEL 3 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(HIPSYCL_DEBUG_LEVEL 1 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

# Make sure either hcc or nvcc can be found
find_package(CUDA)
find_program(HCC_COMPILER NAMES hcc)
find_program(CLANG_EXECUTABLE_PATH NAMES clang++-8 clang++-8.0 clang++-7.0 clang++-7 clang++-6.0 clang++-6 clang++ CACHE STRING)
if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
  message(SEND_ERROR "Could not find clang executable in PATH")
endif()

# Even if we compile with clang, this tests
# for the existence of CUDA which is also required for clang.
# TODO: A better approach would be to find_package(CUDA) if
# we build with clang.
#if(NVCC_COMPILER MATCHES "-NOTFOUND")
#  set(CUDA_FOUND false)
#else()
#  set(CUDA_FOUND true)
#endif()

if(HCC_COMPILER MATCHES "-NOTFOUND")
  set(ROCM_FOUND false)
else()
  set(ROCM_FOUND true)
endif()

if(WITH_CUDA_BACKEND)
  if(NOT CUDA_FOUND)
    message(SEND_ERROR "CUDA was not found")
  endif()
endif()
if(WITH_ROCM_BACKEND)
  if(NOT ROCM_FOUND)
    message(SEND_ERROR "hcc was not found in PATH")
  endif()
endif()

if(CUDA_FOUND)
  set(USE_NVCC false CACHE BOOL "Build hipSYCL with nvcc instead of clang as CUDA compiler (not recommended!)")
endif()

set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
set(WITH_ROCM_BACKEND ${ROCM_FOUND} CACHE BOOL "Build hipSYCL support for AMD GPUs with ROCm")
set(WITH_CPU_BACKEND true CACHE BOOL "Build hipSYCL with support for host execution on CPUs")

if(WITH_CUDA_BACKEND)
  set(DEFAULT_PLATFORM "cuda")
elseif(WITH_ROCM_BACKEND)
  set(DEFAULT_PLATFORM "rocm")
else()
  set(DEFAULT_PLATFORM "cpu")
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})


add_subdirectory(src)

set(DEFAULT_GPU_ARCH "" CACHE STRING "Optional: Default GPU architecture to compile for when targeting GPUs (e.g.: sm_60 or gfx900)")

set(SYCLCC_CONFIG_FILE "{
  \"default-clang\" : \"${CLANG_EXECUTABLE_PATH}\",
  \"default-platform\" : \"${DEFAULT_PLATFORM}\",
  \"default-cuda-path\" : \"${CUDA_TOOLKIT_ROOT_DIR}\",
  \"default-gpu-arch\" : \"${DEFAULT_GPU_ARCH}\",
  \"default-cpu-cxx\" : \"${CMAKE_CXX_COMPILER}\",
  \"default-rocm-path\" : \"/opt/rocm/\",
  \"default-use-bootstrap-mode\" : \"false\",
  \"default-is-dryrun\" : \"false\"
}
")

file(WRITE "${PROJECT_BINARY_DIR}/syclcc.json" ${SYCLCC_CONFIG_FILE})


install(DIRECTORY include/CL DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY contrib/hipCPU/include/hipCPU DESTINATION include/hipSYCL/)

install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/)

install(PROGRAMS bin/syclcc DESTINATION bin)
install(PROGRAMS bin/syclcc-clang DESTINATION bin)

install(FILES ${PROJECT_BINARY_DIR}/syclcc.json DESTINATION etc/hipSYCL/)
