add_executable(platform_api api_test.cpp)
add_sycl_to_target(TARGET platform_api SOURCES api_test.cpp)
