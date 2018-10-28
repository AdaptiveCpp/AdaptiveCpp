add_executable(bruteforce_nbody bruteforce_nbody.cpp)
target_link_libraries(bruteforce_nbody hipSYCL)
install(TARGETS bruteforce_nbody
        RUNTIME DESTINATION share/hipSYCL/examples/)
