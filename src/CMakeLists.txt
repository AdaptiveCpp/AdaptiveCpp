
if(COMPILE_SOURCE_TRANSFORMATION_TOOLS)
  add_subdirectory(hipsycl_rewrite_includes)
  add_subdirectory(hipsycl_transform_source)
ENDIF()
add_subdirectory(compiler)

project(libhipSYCL)

set(CMAKE_CXX_COMPILER ${HIPSYCL_DEVICE_COMPILER})

set(HIPSYCL_SOURCES
  sycl/device.cpp
  sycl/device_selector.cpp
  sycl/exception.cpp
  sycl/queue.cpp
  sycl/handler.cpp
  sycl/local_memory.cpp
  runtime/application.cpp
  runtime/error.cpp
  runtime/backend.cpp
  runtime/hints.cpp
  runtime/device_id.cpp
  runtime/operations.cpp
  runtime/data.cpp
  runtime/multi_queue_executor.cpp
  runtime/dag.cpp
  runtime/dag_node.cpp
  runtime/dag_builder.cpp
  runtime/dag_expander.cpp
  runtime/dag_enumerator.cpp
  runtime/dag_interpreter.cpp
  runtime/dag_scheduler.cpp
  runtime/dag_time_table.cpp
  runtime/dag_manager.cpp
  runtime/dag_submitted_ops.cpp
  runtime/generic/async_worker.cpp
  runtime/hip/hip_error.cpp
  runtime/hip/hip_event.cpp
  runtime/hip/hip_queue.cpp
  runtime/hip/hip_allocator.cpp
  runtime/hip/hip_device_manager.cpp
  runtime/omp/omp_allocator.cpp
  runtime/omp/omp_backend.cpp
  runtime/omp/omp_event.cpp
  runtime/omp/omp_hardware_manager.cpp
  runtime/omp/omp_queue.cpp
  runtime/hw_model/memcpy.cpp)



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
  find_package(OpenMP REQUIRED)
  if(APPLE)
    if(CMAKE_VERSION VERSION_LESS "3.16")
      message(FATAL_ERROR "CMake 3.16.0+ is required for macOS OpenMP support!")
    endif()
  endif()
  add_library(hipSYCL_cpu SHARED ${HIPSYCL_SOURCES})
  target_compile_definitions(hipSYCL_cpu PRIVATE HIPSYCL_BOOTSTRAP=1)
  target_compile_options(hipSYCL_cpu PRIVATE --hipsycl-platform=cpu ${COMMON_SYCLCC_OPTIONS})
  target_include_directories(hipSYCL_cpu PRIVATE ${INCLUDE_DIRS} ${HIPSYCL_SOURCE_DIR}/contrib/hipCPU/include)
  if(APPLE)
    target_include_directories(hipSYCL_cpu PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
  endif()

  target_link_libraries(hipSYCL_cpu PRIVATE --hipsycl-platform=cpu ${COMMON_SYCLCC_OPTIONS})
  install(TARGETS hipSYCL_cpu
          EXPORT install_exports
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
endif()



