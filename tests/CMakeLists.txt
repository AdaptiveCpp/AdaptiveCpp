project(hipsycl-tests)
cmake_minimum_required(VERSION 3.5)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

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
message(STATUS "Found hipSYCL in ${hipSYCL_DIR}")

if(NOT CMAKE_BUILD_TYPE)
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

if(CMAKE_GENERATOR STREQUAL "Ninja" AND
    ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9) OR
    (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)))
  # Force colored warnings in Ninja's output, if the compiler has -fdiagnostics-color support.
  # Rationale in https://github.com/ninja-build/ninja/issues/814
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})


add_executable(device_compilation_tests device_compilation_tests.cpp)
add_sycl_to_target(TARGET device_compilation_tests SOURCES device_compilation_tests.cpp)
target_include_directories(device_compilation_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(device_compilation_tests PRIVATE ${Boost_LIBRARIES})

add_executable(unit_tests unit_tests.cpp)
add_sycl_to_target(TARGET unit_tests SOURCES unit_tests.cpp)
target_include_directories(unit_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unit_tests PRIVATE ${Boost_LIBRARIES})

add_executable(math_unit_tests math_unit_tests.cpp)
add_sycl_to_target(TARGET math_unit_tests SOURCES math_unit_tests.cpp)
target_include_directories(math_unit_tests PRIVATE ${Boost_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(math_unit_tests PRIVATE ${Boost_LIBRARIES})


add_subdirectory(platform_api)
