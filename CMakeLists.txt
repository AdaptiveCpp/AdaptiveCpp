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
find_program(NVCC_COMPILER NAMES nvcc)
find_program(HCC_COMPILER NAMES hcc)

if(NVCC_COMPILER MATCHES "-NOTFOUND")
  set(CUDA_FOUND false)
else()
  set(CUDA_FOUND true)
endif()

if(HCC_COMPILER MATCHES "-NOTFOUND")
  set(ROCM_FOUND false)
else()
  set(ROCM_FOUND true)
endif()

if(WITH_CUDA_BACKEND)
  if(NOT CUDA_FOUND)
    message(SEND_ERROR "nvcc was not found in PATH")
  endif()
endif()
if(WITH_ROCM_BACKEND)
  if(NOT ROCM_FOUND)
    message(SEND_ERROR "hcc was not found in PATH")
  endif()
endif()

set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
set(WITH_ROCM_BACKEND ${ROCM_FOUND} CACHE BOOL "Build hipSYCL support for AMD GPUs with ROCm")
set(WITH_CPU_BACKEND true CACHE BOOL "Build hipSYCL with support for host execution on CPUs")


#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})


add_subdirectory(src)


install(DIRECTORY include/CL DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY contrib/hipCPU/include/hipCPU DESTINATION include/hipSYCL/)

install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/)
install(PROGRAMS bin/syclcc DESTINATION bin)
