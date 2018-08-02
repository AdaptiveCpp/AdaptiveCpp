

project(SYCU)
# ToDo: Detect hcc
set(CMAKE_CXX_COMPILER ${PROJECT_SOURCE_DIR}/bin/sycucc_nv)

cmake_minimum_required (VERSION 2.6)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/contrib/HIP/include)
subdirs(src)
