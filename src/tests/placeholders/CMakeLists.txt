add_executable(placeholder_accessor placeholder.cpp)
target_link_libraries(placeholder_accessor hipSYCL)
install(TARGETS placeholder_accessor
        RUNTIME DESTINATION share/hipSYCL/tests/)
