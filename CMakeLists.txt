cmake_minimum_required (VERSION 3.5)
project(hipSYCL)

set(HIPSYCL_DEVICE_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc-clang)
set(HIPSYCL_SOURCE_DIR ${PROJECT_SOURCE_DIR})

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

if(DISABLE_LLVM_VERSION_CHECK)
  # just use whatever we are given - some llvm distributions do
  # not correctly advertise their versions, and if users explicitly point
  # us to them, let's assume they know what they are doing.
  find_package(LLVM CONFIG)
else()
  # By default, try to find the newest supported LLVM version
  find_package(LLVM 10 CONFIG QUIET)
  if(NOT LLVM_FOUND)
    message(STATUS "No suitable LLVM 10 installation found.")
    find_package(LLVM 9 CONFIG QUIET)
    if(NOT LLVM_FOUND)
      message(STATUS "No suitable LLVM 9 installation found.")
      find_package(LLVM 8 CONFIG QUIET)
    endif()
  endif()
  if(NOT LLVM_FOUND)
    message(SEND_ERROR "Could not find an LLVM installation of version 8 or newer. 
      If you are sure to have passed a suitable LLVM path with -DLLVM_DIR, try again with 
      -DDISABLE_LLVM_VERSION_CHECK=ON.")
  endif()
endif()
message(STATUS "Building hipSYCL against LLVM configured from ${LLVM_DIR}")
#find_package(Clang REQUIRED)

# Check for CUDA/ROCm and clang
find_package(CUDA QUIET)
# We currently search for hipcc to check for ROCm installation
find_program(HIPCC_COMPILER NAMES hipcc HINTS ${ROCM_PATH})
set(ROCM_PATH /opt/rocm CACHE PATH "Path to ROCm installation")

find_program(CLANG_EXECUTABLE_PATH NAMES clang++-${LLVM_VERSION_MAJOR} clang++-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} clang++ CACHE STRING)
if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
  message(SEND_ERROR "Could not find clang executable")
endif()
message(STATUS "Selecting clang: ${CLANG_EXECUTABLE_PATH}")

get_filename_component(LLVM_BIN_DIR ${CLANG_EXECUTABLE_PATH} DIRECTORY)
get_filename_component(LLVM_PREFIX_DIR ${LLVM_BIN_DIR} DIRECTORY)
# The path to the internal clang includes is currently required on ROCm
# to let syclcc-clang fix a wrong order of system includes (clang's internal 
# includes are not of high enough priority in the include path search order).
# We identify this path as the one containing __clang_cuda_runtime_wrapper.h,
# which is a clang-specific header file.
find_path(CLANG_INCLUDE_PATH __clang_cuda_runtime_wrapper.h HINTS
  ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
  ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
  ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}/include
  ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
  ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
  ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include
  DOC "Path to internal clang headers. Typically, $LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include")

if(NOT EXISTS ${CLANG_INCLUDE_PATH})
    message(SEND_ERROR "clang include path ${CLANG_INCLUDE_PATH} does not exist. Please provide clang's internal include path manually.")
endif()
message(STATUS "Using clang include directory: ${CLANG_INCLUDE_PATH}")

set(COMPILE_SOURCE_TRANSFORMATION_TOOLS false CACHE BOOL "Build optional source-to-source transformation tools for the legacy hipSYCL toolchain")

if(HIPCC_COMPILER MATCHES "-NOTFOUND")
  set(ROCM_FOUND false)
else()
  set(ROCM_FOUND true)
endif()

if(WITH_CUDA_BACKEND)
  if(NOT CUDA_FOUND)
    message(SEND_ERROR "CUDA was not found")
  endif()
endif()
if(WITH_ROCM_BACKEND)
  if(NOT ROCM_FOUND)
    #  message(SEND_ERROR "hipcc was not found")
  
    # User has requested ROCm, but we could not find hipcc.
    # this is not necessarily a reason to abort,
    # since we only need libhip_hcc, the HIP includes,
    # and the ROCm device headers. It could be that we
    # are faced with a minimal/incomplete ROCm installation
    # that could still be enough for us.
    # Let's assume the user knows what he/she is doing.
  endif()
endif()

if(CUDA_FOUND)
  set(USE_NVCC false CACHE BOOL "Build hipSYCL with nvcc instead of clang as CUDA compiler (not recommended!)")
endif()

set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
set(WITH_ROCM_BACKEND ${ROCM_FOUND} CACHE BOOL "Build hipSYCL support for AMD GPUs with ROCm")
set(WITH_CPU_BACKEND true CACHE BOOL "Build hipSYCL with support for host execution on CPUs")

if(WITH_CUDA_BACKEND)
  set(DEFAULT_PLATFORM "cuda")
elseif(WITH_ROCM_BACKEND)
  set(DEFAULT_PLATFORM "rocm")
else()
  set(DEFAULT_PLATFORM "cpu")
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SYCLCC_CONFIG_FILE_PATH "${PROJECT_BINARY_DIR}/syclcc.json")
set(SYCLCC_CONFIG_FILE_GLOBAL_INSTALLATION false CACHE BOOL 
  "Whether to install the syclcc configuration file into a global directory (typically, /etc/hipSYCL). This is generally not recommended.")

if(SYCLCC_CONFIG_FILE_GLOBAL_INSTALLATION)
  set(SYCLCC_CONFIG_FILE_INSTALL_DIR /etc/hipSYCL/)
else()
  set(SYCLCC_CONFIG_FILE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/etc/hipSYCL/)
endif()

add_subdirectory(src)

set(DEFAULT_GPU_ARCH "" CACHE STRING "Optional: Default GPU architecture to compile for when targeting GPUs (e.g.: sm_60 or gfx900)")

set(ROCM_LINK_LINE "-rpath $HIPSYCL_ROCM_LIB_PATH -L$HIPSYCL_ROCM_LIB_PATH -lhip_hcc -lamd_comgr -lhsa-runtime64 -rpath $HIPSYCL_ROCM_PATH/hcc/lib -L$HIPSYCL_ROCM_PATH/hcc/lib -lmcwamp -lhc_am" CACHE STRING "Arguments passed to compiler to link ROCm libraries to SYCL applications")
set(CUDA_LINK_LINE "-rpath $HIPSYCL_CUDA_LIB_PATH -L$HIPSYCL_CUDA_LIB_PATH -lcudart" CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")


set(SYCLCC_CONFIG_FILE "{
  \"default-clang\"     : \"${CLANG_EXECUTABLE_PATH}\",
  \"default-platform\"  : \"${DEFAULT_PLATFORM}\",
  \"default-cuda-path\" : \"${CUDA_TOOLKIT_ROOT_DIR}\",
  \"default-gpu-arch\"  : \"${DEFAULT_GPU_ARCH}\",
  \"default-cpu-cxx\"   : \"${CMAKE_CXX_COMPILER}\",
  \"default-rocm-path\" : \"${ROCM_PATH}\",
  \"default-use-bootstrap-mode\" : \"false\",
  \"default-is-dryrun\" : \"false\",
  \"default-clang-include-path\" : \"${CLANG_INCLUDE_PATH}\",
  \"default-rocm-link-line\" : \"${ROCM_LINK_LINE}\",
  \"default-cuda-link-line\" : \"${CUDA_LINK_LINE}\"
}
")

file(WRITE ${SYCLCC_CONFIG_FILE_PATH} ${SYCLCC_CONFIG_FILE})

install(DIRECTORY include/CL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY contrib/hipCPU/include/hipCPU DESTINATION include/hipSYCL/)
install(DIRECTORY contrib/HIP/include/ DESTINATION include/hipSYCL/)

install(PROGRAMS bin/syclcc DESTINATION bin)
install(PROGRAMS bin/syclcc-clang DESTINATION bin)

install(FILES ${SYCLCC_CONFIG_FILE_PATH} DESTINATION ${SYCLCC_CONFIG_FILE_INSTALL_DIR})

set(HIPSYCL_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
configure_file(${PROJECT_SOURCE_DIR}/cmake/hipsycl-config.cmake.in
               ${CMAKE_BINARY_DIR}/hipsycl-config.cmake
               @ONLY)
install(FILES
        ${PROJECT_SOURCE_DIR}/cmake/CMakeDetermineSYCLCompiler.cmake
        ${PROJECT_SOURCE_DIR}/cmake/CMakeSYCLCompiler.cmake.in
        ${PROJECT_SOURCE_DIR}/cmake/CMakeSYCLInformation.cmake
        ${PROJECT_SOURCE_DIR}/cmake/CMakeTestSYCLCompiler.cmake
        ${CMAKE_BINARY_DIR}/hipsycl-config.cmake
        DESTINATION lib/cmake)
install(EXPORT install_exports
        FILE "hipsycl-targets.cmake"
        NAMESPACE hipSYCL::
        DESTINATION lib/cmake)

