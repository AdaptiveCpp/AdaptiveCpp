

project(hipSYCL)
# ToDo: Detect hcc
set(CMAKE_CXX_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc)

cmake_minimum_required (VERSION 2.6)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/contrib/HIP/include)
subdirs(src)

install(DIRECTORY include/CL DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/HIP)
install(PROGRAMS bin/syclcc DESTINATION bin)
