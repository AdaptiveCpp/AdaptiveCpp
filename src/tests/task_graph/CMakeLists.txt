
add_executable(task_graph task_graph.cpp)
target_link_libraries(task_graph sycu)
install(TARGETS task_graph
        RUNTIME DESTINATION share/sycu/tests/)
