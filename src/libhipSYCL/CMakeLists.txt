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
  async_worker.cpp)


set(INCLUDE_DIRS
  ${HIPSYCL_SOURCE_DIR}/include
  ${HIPSYCL_SOURCE_DIR}/contrib/hipCPU/include)

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

  target_compile_options(hipSYCL_cuda PRIVATE --hipsycl-platform=${CUDA_PLATFORM} --hipsycl-bootstrap)
  target_include_directories(hipSYCL_cuda PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(hipSYCL_cuda --hipsycl-platform=${CUDA_PLATFORM} --hipsycl-bootstrap)
  install(TARGETS hipSYCL_cuda
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
endif()

if(WITH_ROCM_BACKEND)
  add_library(hipSYCL_rocm SHARED ${HIPSYCL_SOURCES})
  target_compile_options(hipSYCL_rocm PRIVATE --hipsycl-platform=rocm --hipsycl-bootstrap)
  target_include_directories(hipSYCL_rocm PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(hipSYCL_rocm --hipsycl-platform=rocm --hipsycl-bootstrap)
  install(TARGETS hipSYCL_rocm
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
endif()

if(WITH_CPU_BACKEND)
  add_library(hipSYCL_cpu SHARED ${HIPSYCL_SOURCES})
  target_compile_options(hipSYCL_cpu PRIVATE --hipsycl-platform=cpu --hipsycl-bootstrap)
  target_include_directories(hipSYCL_cpu PRIVATE ${INCLUDE_DIRS})
  target_link_libraries(hipSYCL_cpu --hipsycl-platform=cpu --hipsycl-bootstrap)
  install(TARGETS hipSYCL_cpu
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib/static)
        
endif()
