
add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world sycu)
install(TARGETS hello_world
        RUNTIME DESTINATION share/sycu/tests/)
