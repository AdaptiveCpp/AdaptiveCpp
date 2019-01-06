
set(HIPSYCL_SOURCES 
  device.cpp
  device_selector.cpp
  exception.cpp
  queue.cpp
  handler.cpp
  buffer.cpp
  task_graph.cpp
  accessor.cpp
  async_worker.cpp)

if(WITH_CUDA_BACKEND)

add_library(hipSYCL_cuda SHARED ${HIPSYCL_SOURCES})
target_compile_options(hipSYCL_cuda PRIVATE --hipsycl-platform=cuda --hipsycl-bootstrap)
target_link_libraries(hipSYCL_cuda --hipsycl-platform=cuda --hipsycl-bootstrap)
install(TARGETS hipSYCL_cuda
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
        

endif()

if(WITH_ROCM_BACKEND)

add_library(hipSYCL_rocm SHARED ${HIPSYCL_SOURCES})
target_compile_options(hipSYCL_rocm PRIVATE --hipsycl-platform=rocm --hipsycl-bootstrap)
target_link_libraries(hipSYCL_rocm --hipsycl-platform=rocm --hipsycl-bootstrap)
install(TARGETS hipSYCL_rocm
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
        

endif()

if(WITH_CPU_BACKEND)

add_library(hipSYCL_cpu SHARED ${HIPSYCL_SOURCES})
target_compile_options(hipSYCL_cpu PRIVATE --hipsycl-platform=cpu --hipsycl-bootstrap)
target_link_libraries(hipSYCL_cpu --hipsycl-platform=cpu --hipsycl-bootstrap)
install(TARGETS hipSYCL_cpu
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
        

endif()
