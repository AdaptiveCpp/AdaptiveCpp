cmake_minimum_required (VERSION 3.5)
project(hipSYCL)

set(HIPSYCL_DEVICE_COMPILER ${PROJECT_SOURCE_DIR}/bin/syclcc-clang)
set(HIPSYCL_SOURCE_DIR ${PROJECT_SOURCE_DIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT HIPSYCL_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(HIPSYCL_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(HIPSYCL_DEBUG_LEVEL 1 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

set(ENV{BOOST_ROOT} ${BOOST_PATH})
find_package(Boost COMPONENTS context fiber REQUIRED)
# Check for CUDA/ROCm and clang
find_package(CUDA QUIET)
# We currently search for hipcc to check for ROCm installation
find_program(HIPCC_COMPILER NAMES hipcc HINTS ${ROCM_PATH})
set(ROCM_PATH /opt/rocm CACHE PATH "Path to ROCm installation")

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

set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
set(WITH_ROCM_BACKEND ${ROCM_FOUND} CACHE BOOL "Build hipSYCL support for AMD GPUs with ROCm")

set(WITH_CPU_BACKEND true)

if(WITH_CUDA_BACKEND)
  set(DEFAULT_PLATFORM "cuda")
elseif(WITH_ROCM_BACKEND)
  set(DEFAULT_PLATFORM "rocm")
else()
  set(DEFAULT_PLATFORM "cpu")
endif()

if(WITH_CUDA_BACKEND OR WITH_ROCM_BACKEND)
  set(BUILD_CLANG_PLUGIN true)
endif()


if(BUILD_CLANG_PLUGIN)
  find_package(LLVM CONFIG)
  message(STATUS "Building hipSYCL against LLVM configured from ${LLVM_DIR}")
  #find_package(Clang REQUIRED)

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
  find_path(FOUND_CLANG_INCLUDE_PATH __clang_cuda_runtime_wrapper.h HINTS
    ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
    ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
    ${LLVM_PREFIX_DIR}/include/clang/${LLVM_VERSION_MAJOR}/include
    ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
    ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}/include
    ${LLVM_PREFIX_DIR}/lib/clang/${LLVM_VERSION_MAJOR}/include
    DOC "Path to internal clang headers. Typically, $LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include")

  if(EXISTS ${FOUND_CLANG_INCLUDE_PATH})
    # Required for newer ROCm versions
    set(CLANG_INCLUDE_PATH ${FOUND_CLANG_INCLUDE_PATH}/..)
  endif()
  
  if(NOT EXISTS ${CLANG_INCLUDE_PATH})
    message(SEND_ERROR "clang include path ${CLANG_INCLUDE_PATH} does not exist. Please provide clang's internal include path manually: Find the directory where __clang_cuda_runtime_wrapper.h is. Provide this directory for older ROCm versions and the parent directory for newer ones.")
  endif()
  message(STATUS "Using clang include directory: ${CLANG_INCLUDE_PATH}")
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

set(DEFAULT_GPU_ARCH "" CACHE STRING "Optional: Default GPU architecture to compile for when targeting GPUs (e.g.: sm_60 or gfx900)")
set(ROCM_LIBS "-L${ROCM_PATH}/lib -L${ROCM_PATH}/hip/lib -lamdhip64" CACHE STRING "Necessary libraries for ROCm")
if(NOT ROCM_LINK_LINE)
  set(ROCM_LINK_LINE "-rpath $HIPSYCL_ROCM_PATH/lib -Wl,-rpath=$HIPSYCL_ROCM_PATH/hip/lib ${ROCM_LIBS}")
endif()
if(NOT ROCM_CXX_FLAGS)
  # clang erroneously sets feature detection flags for 
  # __float128 even though it is not supported for CUDA / HIP,
  # see https://bugs.llvm.org/show_bug.cgi?id=47559.
  set(ROCM_CXX_FLAGS "-isystem ${CLANG_INCLUDE_PATH} -U__FLOAT128__ -U__SIZEOF_FLOAT128__ -I$HIPSYCL_ROCM_PATH/include -I$HIPSYCL_ROCM_PATH/include --hip-device-lib-path=$HIPSYCL_ROCM_PATH/lib ")
endif()
if(NOT CUDA_LINK_LINE) 
  set(CUDA_LINK_LINE "-rpath $HIPSYCL_CUDA_LIB_PATH -L$HIPSYCL_CUDA_LIB_PATH -lcudart" CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
endif()
if(NOT CUDA_CXX_FLAGS)	
  # clang erroneously sets feature detection flags for 
  # __float128 even though it is not supported for CUDA / HIP,
  # see https://bugs.llvm.org/show_bug.cgi?id=47559.
  set(CUDA_CXX_FLAGS "-U__FLOAT128__ -U__SIZEOF_FLOAT128__")
endif()
if(NOT OMP_LINK_LINE)
  set(OMP_LINK_LINE "-L${BOOST_PATH}/lib -lboost_context -lboost_fiber -Wl,-rpath=${BOOST_PATH}/lib -fopenmp")
endif()
if(NOT OMP_CXX_FLAGS) 
  set(OMP_CXX_FLAGS "-I${BOOST_PATH}/include -fopenmp")
endif()
if(NOT SEQUENTIAL_LINK_LINE) 
  set(SEQUENTIAL_LINK_LINE "-L${BOOST_PATH}/lib -lboost_context -lboost_fiber -lomp  -Wl,-rpath=${BOOST_PATH}/lib")
endif()
if(NOT SEQUENTIAL_CXX_FLAGS) 
  set(SEQUENTIAL_CXX_FLAGS "-I${BOOST_PATH}/include")
endif()
add_subdirectory(src)

set(SYCLCC_CONFIG_FILE "{
  \"default-clang\"     : \"${CLANG_EXECUTABLE_PATH}\",
  \"default-platform\"  : \"${DEFAULT_PLATFORM}\",
  \"default-cuda-path\" : \"${CUDA_TOOLKIT_ROOT_DIR}\",
  \"default-gpu-arch\"  : \"${DEFAULT_GPU_ARCH}\",
  \"default-cpu-cxx\"   : \"${CMAKE_CXX_COMPILER}\",
  \"default-rocm-path\" : \"${ROCM_PATH}\",
  \"default-boost-path\" : \"${BOOST_PATH}\",
  \"default-use-bootstrap-mode\" : \"false\",
  \"default-is-dryrun\" : \"false\",
  \"default-clang-include-path\" : \"${CLANG_INCLUDE_PATH}\",
  \"default-sequential-link-line\" : \"${SEQUENTIAL_LINK_LINE}\",
  \"default-sequential-cxx-flags\" : \"${SEQUENTIAL_CXX_FLAGS}\",
  \"default-omp-link-line\" : \"${OMP_LINK_LINE}\",
  \"default-omp-cxx-flags\" : \"${OMP_CXX_FLAGS}\",
  \"default-rocm-link-line\" : \"${ROCM_LINK_LINE}\",
  \"default-rocm-cxx-flags\" : \"${ROCM_CXX_FLAGS}\",
  \"default-cuda-link-line\" : \"${CUDA_LINK_LINE}\",
  \"default-cuda-cxx-flags\" : \"${CUDA_CXX_FLAGS}\"
}
")

file(WRITE ${SYCLCC_CONFIG_FILE_PATH} ${SYCLCC_CONFIG_FILE})

install(DIRECTORY include/CL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/SYCL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/hipSYCL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")

install(PROGRAMS bin/syclcc DESTINATION bin)
install(PROGRAMS bin/syclcc-clang DESTINATION bin)
install(PROGRAMS cmake/syclcc-launcher DESTINATION lib/cmake/hipSYCL)
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION lib/cmake/hipSYCL)

install(FILES ${SYCLCC_CONFIG_FILE_PATH} DESTINATION ${SYCLCC_CONFIG_FILE_INSTALL_DIR})

set(HIPSYCL_INSTALL_LOCATION ${CMAKE_INSTALL_PREFIX})
configure_file(${PROJECT_SOURCE_DIR}/cmake/hipsycl-config.cmake.in
               ${CMAKE_BINARY_DIR}/hipsycl-config.cmake
               @ONLY)
install(FILES
        ${CMAKE_BINARY_DIR}/hipsycl-config.cmake
        DESTINATION lib/cmake/hipSYCL)
install(EXPORT install_exports
        FILE "hipsycl-targets.cmake"
        NAMESPACE hipSYCL::
        DESTINATION lib/cmake/hipSYCL)

