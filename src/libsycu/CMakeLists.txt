
add_library(sycu SHARED
  device.cpp
  device_selector.cpp
  exception.cpp
  queue.cpp
  handler.cpp
  buffer.cpp
  task_graph.cpp
  accessor.cpp)

install(TARGETS sycu
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
