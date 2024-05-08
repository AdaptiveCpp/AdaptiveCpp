add_executable(dump_test dump_test.cpp)
add_sycl_to_target(TARGET dump_test)
target_include_directories(dump_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenMP_CXX_INCLUDE_DIRS})
