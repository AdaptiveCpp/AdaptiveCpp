
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


add_library(hipSYCL-rt SHARED
  application.cpp
  runtime.cpp
  error.cpp
  backend.cpp
  backend_loader.cpp
  hints.cpp
  device_id.cpp
  operations.cpp
  data.cpp
  multi_queue_executor.cpp
  dag.cpp
  dag_node.cpp
  dag_builder.cpp
  dag_direct_scheduler.cpp
  dag_manager.cpp
  dag_submitted_ops.cpp
  generic/async_worker.cpp
  hw_model/memcpy.cpp
  serialization/serialization.cpp)

# syclcc already knows about these include directories, but clangd-based tooling does not.
# Specifying them explicitly ensures that IDEs can resolve all hipSYCL includes correctly.
target_include_directories(hipSYCL-rt
  PUBLIC
    $<BUILD_INTERFACE:${HIPSYCL_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${HIPSYCL_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include
)

install(TARGETS hipSYCL-rt
        EXPORT install_exports
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)


if(WITH_CUDA_BACKEND)
  add_library(rt-backend-cuda SHARED
    cuda/cuda_event.cpp
    cuda/cuda_queue.cpp
    cuda/cuda_allocator.cpp
    cuda/cuda_device_manager.cpp
    cuda/cuda_hardware_manager.cpp
    cuda/cuda_backend.cpp
    cuda/cuda_module.cpp)

  target_include_directories(rt-backend-cuda PRIVATE
    ${HIPSYCL_SOURCE_DIR}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include)
  
  target_link_libraries(rt-backend-cuda PRIVATE hipSYCL-rt ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

  install(TARGETS rt-backend-cuda
        LIBRARY DESTINATION lib/hipSYCL
        ARCHIVE DESTINATION lib/hipSYCL)
endif()

if(WITH_ROCM_BACKEND)
  add_library(rt-backend-hip SHARED
    hip/hip_event.cpp
    hip/hip_queue.cpp
    hip/hip_allocator.cpp
    hip/hip_device_manager.cpp
    hip/hip_hardware_manager.cpp
    hip/hip_backend.cpp)

  target_compile_definitions(rt-backend-hip PRIVATE HIPSYCL_RT_HIP_TARGET_ROCM=1)
  target_include_directories(rt-backend-hip PRIVATE ${HIPSYCL_SOURCE_DIR}/include)
  if(NOT HIP_FOUND)
    target_include_directories(rt-backend-hip PRIVATE ${ROCM_PATH}/include)
    target_link_libraries(rt-backend-hip PRIVATE ${ROCM_LIBS})
  else()
    # Supress warnings because wrongly set CXX arguments
    target_compile_options(rt-backend-hip PRIVATE -Wno-unused-command-line-argument)
    target_link_libraries(rt-backend-hip PRIVATE hipSYCL-rt  hip::host)
  endif()

  install(TARGETS rt-backend-hip
        LIBRARY DESTINATION lib/hipSYCL
        ARCHIVE DESTINATION lib/hipSYCL)
endif()

if(WITH_LEVEL_ZERO_BACKEND)
  add_library(rt-backend-ze SHARED
    ze/ze_backend.cpp
    ze/ze_hardware_manager.cpp
    ze/ze_allocator.cpp
    ze/ze_event.cpp
    ze/ze_queue.cpp)
  
  target_include_directories(rt-backend-ze PRIVATE ${HIPSYCL_SOURCE_DIR}/include)
  target_link_libraries(rt-backend-ze PRIVATE hipSYCL-rt -lze_loader)
endif()

if(WITH_CPU_BACKEND)
  add_library(rt-backend-omp SHARED
    omp/omp_allocator.cpp
    omp/omp_backend.cpp
    omp/omp_event.cpp
    omp/omp_hardware_manager.cpp
    omp/omp_queue.cpp)

    find_package(OpenMP REQUIRED)

    target_include_directories(rt-backend-omp PRIVATE ${HIPSYCL_SOURCE_DIR}/include)
    if(APPLE)
      if(CMAKE_VERSION VERSION_LESS "3.16")
        message(FATAL_ERROR "CMake 3.16.0+ is required for macOS OpenMP support!")
      endif()
      target_include_directories(rt-backend-omp PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    endif()

    target_link_libraries(rt-backend-omp PRIVATE hipSYCL-rt  OpenMP::OpenMP_CXX)

    install(TARGETS rt-backend-omp
        LIBRARY DESTINATION lib/hipSYCL
        ARCHIVE DESTINATION lib/hipSYCL)
endif()

