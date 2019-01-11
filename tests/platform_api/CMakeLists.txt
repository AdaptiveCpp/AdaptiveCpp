

add_executable(platform_api api_test.cpp)
target_link_libraries(platform_api)
install(TARGETS platform_api
        RUNTIME DESTINATION share/hipSYCL/tests/)
