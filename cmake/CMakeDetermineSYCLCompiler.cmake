set(CMAKE_SYCL_COMPILER ${HIPSYCL_SYCLCC})

# We could set this, but prefer to set the LANGUAGE per source file
# individiually in order to avoid conflicts with pure CXX executables.
# set(CMAKE_SYCL_SOURCE_FILE_EXTENSIONS ${CMAKE_CXX_SOURCE_FILE_EXTENSIONS})

set(CMAKE_SYCL_OUTPUT_EXTENSION .o)
set(CMAKE_SYCL_COMPILER_ENV_VAR "SYCL")
set(CMAKE_SYCL_LINKER_PREFERENCE "CXX")

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeSYCLCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeSYCLCompiler.cmake
               @ONLY)

