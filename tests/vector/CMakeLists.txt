add_executable(vector vec.cpp)
target_link_libraries(vector)
install(TARGETS vector
        RUNTIME DESTINATION share/hipSYCL/tests/)
