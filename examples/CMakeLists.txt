project(hipsycl-examples)

if(HIPSYCL_INSTALL_DIR)
  # We make sure that we don't override an already set hipSYCL_DIR variable
  if (hipSYCL_DIR)
    message(FATAL_ERROR "Please only use HIPSYCL_INSTALL_DIR or hipSYCL_DIR")
  endif() 
  set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake)
  # We don't set the PATHS var here since we don't want to find something other than the user intends to.
  find_package(hipSYCL CONFIG REQUIRED)
else()
  # The user might have set hipSYCL_DIR or some other var manually,
  # therefore we first try to find hipSYCL with default settings 
  find_package(hipSYCL CONFIG)
  # If found in the previous step nothing will be searched here
  find_package(hipSYCL CONFIG PATHS /opt/hipSYCL/lib/cmake/ REQUIRED)
endif()
message(STATUS "Found hipSYCL in ${hipSYCL_DIR}")if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT HIPSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(HIPSYCL_DEBUG_LEVEL 3 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(HIPSYCL_DEBUG_LEVEL 1 CACHE INTEGER
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

cmake_minimum_required (VERSION 3.5)

include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})

subdirs(bruteforce_nbody)
