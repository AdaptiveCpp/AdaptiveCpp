add_executable(hierarchical_dispatch hierarchical_dispatch.cpp)
target_link_libraries(hierarchical_dispatch)
install(TARGETS hierarchical_dispatch
        RUNTIME DESTINATION share/hipSYCL/tests/)
