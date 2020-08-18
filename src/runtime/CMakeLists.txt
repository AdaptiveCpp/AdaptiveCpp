

set(HIPSYCL_SOURCES
  application.cpp
  runtime.cpp
  error.cpp
  backend.cpp
  hints.cpp
  device_id.cpp
  operations.cpp
  data.cpp
  multi_queue_executor.cpp
  dag.cpp
  dag_node.cpp
  dag_builder.cpp
  dag_expander.cpp
  dag_enumerator.cpp
  dag_interpreter.cpp
  dag_scheduler.cpp
  dag_direct_scheduler.cpp
  dag_time_table.cpp
  dag_manager.cpp
  dag_submitted_ops.cpp
  generic/async_worker.cpp
  hw_model/memcpy.cpp
  serialization/serialization.cpp)

if(WITH_CUDA_BACKEND)
  set(HIPSYCL_SOURCES ${HIPSYCL_SOURCES}
    cuda/cuda_event.cpp
    cuda/cuda_queue.cpp
    cuda/cuda_allocator.cpp
    cuda/cuda_device_manager.cpp
    cuda/cuda_hardware_manager.cpp
    cuda/cuda_backend.cpp)
endif()

if(WITH_ROCM_BACKEND)
  set(HIPSYCL_SOURCES ${HIPSYCL_SOURCES}
    hip/hip_event.cpp
    hip/hip_queue.cpp
    hip/hip_allocator.cpp
    hip/hip_device_manager.cpp
    hip/hip_hardware_manager.cpp
    hip/hip_backend.cpp)
endif()

if(WITH_CPU_BACKEND)
  set(HIPSYCL_SOURCES ${HIPSYCL_SOURCES}
    omp/omp_allocator.cpp
    omp/omp_backend.cpp
    omp/omp_event.cpp
    omp/omp_hardware_manager.cpp
    omp/omp_queue.cpp)
endif()

set(INCLUDE_DIRS
  ${HIPSYCL_SOURCE_DIR}/include
  ${HIPSYCL_SOURCE_DIR})

add_library(hipSYCL-rt SHARED ${HIPSYCL_SOURCES})
target_include_directories(hipSYCL-rt PRIVATE ${INCLUDE_DIRS})

if(WITH_CUDA_BACKEND)
  target_compile_definitions(hipSYCL-rt PRIVATE HIPSYCL_RT_ENABLE_CUDA_BACKEND=1)
  target_include_directories(hipSYCL-rt PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/include)
  target_link_libraries(hipSYCL-rt PRIVATE ${CUDA_LIBRARIES})
endif()

if(WITH_ROCM_BACKEND) 
  target_compile_definitions(hipSYCL-rt PRIVATE HIPSYCL_RT_ENABLE_HIP_BACKEND=1 HIPSYCL_RT_HIP_TARGET_ROCM=1)
  target_include_directories(hipSYCL-rt PRIVATE ${ROCM_PATH}/include)
endif()

if(WITH_CPU_BACKEND)
  find_package(OpenMP REQUIRED)
  if(APPLE)
    if(CMAKE_VERSION VERSION_LESS "3.16")
      message(FATAL_ERROR "CMake 3.16.0+ is required for macOS OpenMP support!")
    endif()
    target_include_directories(hipSYCL-rt PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
  endif()

  target_link_libraries(hipSYCL-rt PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(hipSYCL-rt PRIVATE HIPSYCL_RT_ENABLE_OMP_BACKEND=1)
endif()

install(TARGETS hipSYCL-rt
        EXPORT install_exports
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)



