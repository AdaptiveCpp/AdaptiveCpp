add_executable(vector_add vector_add.cpp)
target_link_libraries(vector_add hipSYCL)
install(TARGETS vector_add
        RUNTIME DESTINATION share/hipSYCL/tests/)
