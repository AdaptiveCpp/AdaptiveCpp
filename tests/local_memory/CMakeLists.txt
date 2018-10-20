add_executable(local_memory local_memory.cpp)
target_link_libraries(local_memory hipSYCL)
install(TARGETS local_memory
        RUNTIME DESTINATION share/hipSYCL/tests/)
