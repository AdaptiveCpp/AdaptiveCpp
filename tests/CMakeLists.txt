project(hipsycl-tests)
cmake_minimum_required(VERSION 3.5)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

find_program(SYCLCC NAMES syclcc CACHE STRING)
if(SYCLCC MATCHES SYCLCC-NOTFOUND)
  message(SEND_ERROR "hipSYCL syclcc compiler not found, exiting.")
  return()
endif()

set(CMAKE_CXX_COMPILER ${SYCLCC})
set(CMAKE_CXX_FLAGS "--restrict-device-header-path=")

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

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

add_subdirectory(platform_api)

add_executable(unit_tests unit_tests.cpp)
target_include_directories(unit_tests PRIVATE ${Boost_HEADER_FILES} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unit_tests PRIVATE ${Boost_LIBRARIES})

