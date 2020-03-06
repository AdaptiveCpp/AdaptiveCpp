project(libhipSYCL)

set(CMAKE_CXX_COMPILER ${HIPSYCL_DEVICE_COMPILER})

set(HIPSYCL_SOURCES
  application.cpp
  device.cpp
  device_selector.cpp
  exception.cpp
  queue.cpp
  handler.cpp
  buffer.cpp
  task_graph.cpp
  accessor.cpp
  async_worker.cpp
  local_memory.cpp)


set(INCLUDE_DIRS
  ${HIPSYCL_SOURCE_DIR}/include
  ${HIPSYCL_SOURCE_DIR})

set(COMMON_SYCLCC_OPTIONS --hipsycl-bootstrap --hipsycl-config-file=${SYCLCC_CONFIG_FILE_PATH})

# Specify GPU architectures used when compiling libhipSYCL. 
# They do not really matter as libhipSYCL does not
# contain any kernel code, and are only here to make syclcc happy since it will complain if
# there's no GPU arch specified.
set(HIPSYCL_CUDA_DUMMY_GPU_ARCH sm_52)
set(HIPSYCL_ROCM_DUMMY_GPU_ARCH gfx900)


if(WITH_CUDA_BACKEND)
  add_library(hipSYCL_cuda SHARED ${HIPSYCL_SOURCES})

  if(USE_NVCC)
      if(NOT CMAKE_CUDA_STANDARD)
          set (CMAKE_CUDA_STANDARD 14)
      endif()
    set(CUDA_PLATFORM nvcc)
    target_compile_options(hipSYCL_cuda PRIVATE --std c++${CMAKE_CUDA_STANDARD})
    if(CMAKE_CUDA_HOST_COMPILER)
        target_compile_options(hipSYCL_cuda PRIVATE -ccbin ${CMAKE_CUDA_HOST_COMPILER})
    endif()
  else()
    set(CUDA_PLATFORM cuda)
  endif()

  target_compile_definitions(hipSYCL_cuda PRIVATE HIPSYCL_BOOTSTRAP=1)
  target_compile_options(hipSYCL_cuda PRIVATE --hipsycl-platform=${CUDA_PLATFORM} --hipsycl-gpu-arch=${HIPSYCL_CUDA_DUMMY_GPU_ARCH} ${COMMON_SYCLCC_OPTIONS})
  target_include_directories(hipSYCL_cuda PRIVATE ${INCLUDE_DIRS} ${HIPSYCL_SOURCE_DIR}/contrib/HIP/include)
  target_link_libraries(hipSYCL_cuda PRIVATE --hipsycl-platform=${CUDA_PLATFORM} --hipsycl-gpu-arch=${HIPSYCL_CUDA_DUMMY_GPU_ARCH} ${COMMON_SYCLCC_OPTIONS})
  install(TARGETS hipSYCL_cuda
          EXPORT install_exports
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
endif()

if(WITH_ROCM_BACKEND)
  add_library(hipSYCL_rocm SHARED ${HIPSYCL_SOURCES})
  #Try to peek if ROCM set itself through its /etc/profile.d/rocm.sh
  set(ROCM_PATH $ENV{ROCM_PATH})
  target_compile_definitions(hipSYCL_rocm PRIVATE HIPSYCL_BOOTSTRAP=1)
  target_compile_options(hipSYCL_rocm PRIVATE --hipsycl-platform=rocm --hipsycl-gpu-arch=${HIPSYCL_ROCM_DUMMY_GPU_ARCH} ${COMMON_SYCLCC_OPTIONS})
  target_include_directories(hipSYCL_rocm PRIVATE ${INCLUDE_DIRS})
  if(ROCM_PATH)
      target_include_directories(hipSYCL_rocm PRIVATE ${ROCM_PATH}/include )
  endif()
  target_link_libraries(hipSYCL_rocm PRIVATE --hipsycl-platform=rocm --hipsycl-gpu-arch=${HIPSYCL_ROCM_DUMMY_GPU_ARCH} ${COMMON_SYCLCC_OPTIONS})
  install(TARGETS hipSYCL_rocm
          EXPORT install_exports
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
endif()

if(WITH_CPU_BACKEND)
  add_library(hipSYCL_cpu SHARED ${HIPSYCL_SOURCES})
  target_compile_definitions(hipSYCL_cpu PRIVATE HIPSYCL_BOOTSTRAP=1)
  target_compile_options(hipSYCL_cpu PRIVATE --hipsycl-platform=cpu ${COMMON_SYCLCC_OPTIONS})
  target_include_directories(hipSYCL_cpu PRIVATE ${INCLUDE_DIRS} ${HIPSYCL_SOURCE_DIR}/contrib/hipCPU/include)
  target_link_libraries(hipSYCL_cpu PRIVATE --hipsycl-platform=cpu ${COMMON_SYCLCC_OPTIONS})
  install(TARGETS hipSYCL_cpu
          EXPORT install_exports
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
endif()

