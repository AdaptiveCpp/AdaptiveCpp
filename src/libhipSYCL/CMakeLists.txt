project(libhipSYCL)

set(CMAKE_CXX_COMPILER ${HIPSYCL_DEVICE_COMPILER})

add_library(hipSYCL SHARED
  device.cpp
  device_selector.cpp
  exception.cpp
  queue.cpp
  handler.cpp
  buffer.cpp
  task_graph.cpp
  accessor.cpp
  async_worker.cpp)

target_include_directories(hipSYCL PRIVATE
  ../../include)

install(TARGETS hipSYCL
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
