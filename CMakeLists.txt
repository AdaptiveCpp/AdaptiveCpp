cmake_minimum_required (VERSION 3.5)
project(hipSYCL)

# Make sure either hcc or nvcc can be found
find_program(VENDOR_DEVICE_COMPILER NAMES hcc nvcc)
if(VENDOR_DEVICE_COMPILER MATCHES "-NOTFOUND")
  message(SEND_ERROR "Neither 'hcc' nor 'nvcc' was found in PATH")
endif()
set(HIPSYCL_DEVICE_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc)

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
add_subdirectory(src)

install(DIRECTORY include/CL DESTINATION include/
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/dummy_hip DESTINATION include/
        FILES_MATCHING PATTERN "*.h")

install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/)
install(PROGRAMS bin/syclcc DESTINATION bin)
