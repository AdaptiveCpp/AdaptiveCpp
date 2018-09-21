project(hipSYCL)

set(CMAKE_CXX_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc)

if(NOT HIPSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(HIPSYCL_DEBUG_LEVEL 3 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(HIPSYCL_DEBUG_LEVEL 0 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()
#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

cmake_minimum_required (VERSION 3.5)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/include)
subdirs(src)

install(DIRECTORY include/CL DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/)
install(PROGRAMS bin/syclcc DESTINATION bin)
