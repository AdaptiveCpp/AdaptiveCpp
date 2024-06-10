add_executable(platform_api api_test.cpp)
target_include_directories(platform_api PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
add_sycl_to_target(TARGET platform_api)
