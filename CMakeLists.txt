project(hipSYCL)

set(CMAKE_CXX_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc)

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

if(WITH_CUDA_BACKEND)
  find_package(CUDA REQUIRED)
else()
  find_package(CUDA)
endif()

if(CUDA_FOUND)
  set(WITH_CUDA_BACKEND true CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
else()
  set(WITH_CUDA_BACKEND false CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
endif()

set(WITH_CPU_BACKEND true CACHE BOOL "Build hipSYCL with support for host execution on CPUs")
# TODO Detect hcc
set(WITH_ROCM_BACKEND false CACHE BOOL "Build hipSYCL support for AMD GPUs with ROCm")


#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

cmake_minimum_required (VERSION 3.5)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/contrib/hipCPU/include)
subdirs(src)

install(DIRECTORY include/CL DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY contrib/hipCPU/include/hipCPU DESTINATION include/hipSYCL/)

install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/)
install(PROGRAMS bin/syclcc DESTINATION bin)
