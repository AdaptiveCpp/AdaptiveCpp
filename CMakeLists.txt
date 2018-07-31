

project(SYCU)
# ToDo: Detect hcc
set(CMAKE_CXX_COMPILER nvcc)
set(CMAKE_CXX_FLAGS "-x cu")

cmake_minimum_required (VERSION 2.6)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/contrib/HIP/include)
subdirs(src)
