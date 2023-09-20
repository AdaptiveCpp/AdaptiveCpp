cmake_minimum_required (VERSION 3.5)
if(NOT CMAKE_VERSION VERSION_LESS 3.12)
  cmake_policy(SET CMP0074 NEW) # Don't complain about using BOOST_ROOT...
endif()

set(HIPSYCL_VERSION_MAJOR 23)
set(HIPSYCL_VERSION_MINOR 10)
set(HIPSYCL_VERSION_PATCH 0)

project(hipSYCL VERSION ${HIPSYCL_VERSION_MAJOR}.${HIPSYCL_VERSION_MINOR}.${HIPSYCL_VERSION_PATCH})

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
    set(HIPSYCL_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

if (NOT ${BOOST_ROOT} STREQUAL "")
  set(ENV{BOOST_ROOT} ${BOOST_ROOT})
endif()

find_package(Boost COMPONENTS context fiber )

if(NOT Boost_FOUND)
  message(FATAL_ERROR "Please provide the path to the root directory of the boost installation using the -DBOOST_ROOT option")
endif()

# Check for CUDA/ROCm and clang
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/")
find_package(CUDA QUIET)
find_package(HIP QUIET HINTS ${ROCM_PATH} ${ROCM_PATH}/lib/cmake)
# Check for OpenCL
find_package(OpenCL QUIET)

# In case HIP is not found via the cmake pkgs we fall back to the legacy rocm discovery:
if (NOT HIP_FOUND)
  message(STATUS "Could not find HIP cmake integration, falling back to finding hipcc")
  find_program(HIPCC_COMPILER NAMES hipcc HINTS ${ROCM_PATH})
  set(ROCM_PATH /opt/rocm CACHE PATH "Path to ROCm installation")
  if(HIPCC_COMPILER MATCHES "-NOTFOUND")
    message(STATUS "Could not find ROCm installation by looking for hipcc. ROCm support unavailable.")
    set(ROCM_FOUND false)
  else()
    set(ROCM_FOUND true)
  endif()
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

if(WITH_OPENCL_BACKEND)
  if(NOT OpenCL_FOUND)
    message(SEND_ERROR "OpenCL was not found")
  endif()
endif()

set(WITH_CUDA_BACKEND ${CUDA_FOUND} CACHE BOOL "Build hipSYCL support for NVIDIA GPUs with CUDA")
set(WITH_ROCM_BACKEND ${ROCM_FOUND} CACHE BOOL "Build hipSYCL support for AMD GPUs with ROCm")
set(WITH_OPENCL_BACKEND ${OpenCL_FOUND} CACHE BOOL "Build hipSYCL support for OpenCL SPIR-V devices supporting USM")
set(WITH_CUDA_NVCXX_ONLY FALSE CACHE BOOL "Whether to target CUDA exclusively with nvc++")

if(APPLE OR WIN32 OR WITH_CUDA_NVCXX_ONLY)
  set(WITH_ACCELERATED_CPU false CACHE BOOL "Build hipSYCL compiler support for accelerating the CPU backend")
  set(WITH_SSCP_COMPILER false CACHE BOOL "Build hipSYCL generic SSCP compiler support")
  set(WITH_STDPAR_COMPILER false CACHE BOOL "Build hipSYCL C++ standard parallelism support")
else()
  set(WITH_ACCELERATED_CPU true CACHE BOOL "Build hipSYCL compiler support for accelerating the CPU backend")
  set(WITH_SSCP_COMPILER true CACHE BOOL "Build hipSYCL generic SSCP compiler support")
  set(WITH_STDPAR_COMPILER true CACHE BOOL "Build hipSYCL C++ standard parallelism compiler support")
endif()
set(WITH_CPU_BACKEND true)

if(WITH_CUDA_BACKEND)
  set(DEFAULT_PLATFORM "cuda")

  find_program(NVCXX_COMPILER NAMES nvc++)
elseif(WITH_ROCM_BACKEND)
  set(DEFAULT_PLATFORM "rocm")
else()
  set(DEFAULT_PLATFORM "cpu")
endif()

if(WITH_CUDA_NVCXX_ONLY)
  if(NVCXX_COMPILER MATCHES "-NOTFOUND")
      message(SEND_ERROR "Could not find nvc++ executable, specifiy with -DNVCXX_COMPILER=")
  endif()
endif()


if(WITH_ACCELERATED_CPU) # sanitize input
  set(WITH_ACCELERATED_CPU true)
  set(BUILD_CLANG_PLUGIN true)
else()
  set(WITH_ACCELERATED_CPU false)
endif()

if(WITH_SSCP_COMPILER) # sanitize input
  set(WITH_SSCP_COMPILER true)
  set(BUILD_CLANG_PLUGIN true)
else()
  set(WITH_SSCP_COMPILER false)
endif()

if(WITH_STDPAR_COMPILER) # sanitize input
  set(WITH_STDPAR_COMPILER true)
  set(BUILD_CLANG_PLUGIN true)
else()
  set(WITH_STDPAR_COMPILER false)
endif()

if(WITH_CUDA_BACKEND AND NOT WITH_CUDA_NVCXX_ONLY)
  set(BUILD_CLANG_PLUGIN true)
endif()
if(WITH_ROCM_BACKEND)
  set(BUILD_CLANG_PLUGIN true)
endif()
if(WITH_OPENCL_BACKEND OR WITH_LEVEL_ZERO_BACKEND)
  set(BUILD_CLANG_PLUGIN true)
  set(WITH_SSCP_COMPILER true)
endif()

if(BUILD_CLANG_PLUGIN)
  find_package(LLVM CONFIG)
  message(STATUS "Building hipSYCL against LLVM configured from ${LLVM_DIR}")
  #find_package(Clang REQUIRED)

  find_program(CLANG_EXECUTABLE_PATH NAMES clang++-${LLVM_VERSION_MAJOR} clang++-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} clang++ HINTS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH CACHE STRING)
  if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
    find_program(CLANG_EXECUTABLE_PATH NAMES clang++-${LLVM_VERSION_MAJOR} clang++-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} clang++ CACHE STRING)
    if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
      message(SEND_ERROR "Could not find clang executable")
    endif()
  endif()
  if(WIN32)
    # plugins link agains clang.exe and not clang++.exe, thus using clang++ would result
    # in symbol clashes and therefore the plugin not loading.
    # on Windows it seems to be fine, as long as clang is able to deduce the language
    # from the file extension, as the msvcrt is linked explicitly anyways.
    string(REPLACE "clang++.exe" "clang.exe" CLANG_EXECUTABLE_PATH ${CLANG_EXECUTABLE_PATH})
  endif()
  message(STATUS "Selecting clang: ${CLANG_EXECUTABLE_PATH}")

  get_filename_component(LLVM_BIN_DIR "${CLANG_EXECUTABLE_PATH}" DIRECTORY)
  get_filename_component(LLVM_PREFIX_DIR "${LLVM_BIN_DIR}" DIRECTORY)
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
    ${LLVM_PREFIX_DIR}/lib64/clang/${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}/include
    DOC "Path to internal clang headers. Typically, $LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include")

  if(EXISTS ${FOUND_CLANG_INCLUDE_PATH})
    # Required for newer ROCm versions
    set(CLANG_INCLUDE_PATH ${FOUND_CLANG_INCLUDE_PATH}/..)
  endif()
  
  if(NOT EXISTS ${CLANG_INCLUDE_PATH})
    message(SEND_ERROR "clang include path ${CLANG_INCLUDE_PATH} does not exist. Please provide clang's internal include path manually: Find the directory where __clang_cuda_runtime_wrapper.h is. Provide this directory for older ROCm versions and the parent directory for newer ones.")
  endif()
  if(WITH_ROCM_BACKEND)
    execute_process(COMMAND ${CLANG_EXECUTABLE_PATH} "--version"
      OUTPUT_VARIABLE CLANG_VERBOSE_OUTPUT)
    string(REGEX MATCH "clang version [0-9]+.[0-9]+.[0-9]+ \\(.+ roc-([0-9]+).([0-9]+).([0-9]+)" AMD_CLANG_VERSION ${CLANG_VERBOSE_OUTPUT})

    if(AMD_CLANG_VERSION)
      set(ROCM_VERSION_MAJOR ${CMAKE_MATCH_1})
      set(ROCM_VERSION_MINOR ${CMAKE_MATCH_2})
      set(ROCM_VERSION_PATCH ${CMAKE_MATCH_3})
      message(STATUS "AMD clang version: ${ROCM_VERSION_MAJOR}.${ROCM_VERSION_MINOR}.${ROCM_VERSION_PATCH}")

      if(${LLVM_VERSION_MAJOR} EQUAL 13 AND NOT DEFINED HIPSYCL_NO_DEVICE_MANGLER)
        message(SEND_ERROR "AMD clang version not compatible with hipSYCL explicit multipass. "
                           "To disable explicit multipass set -DHIPSYCL_NO_DEVICE_MANGLER=ON. "
                           "This also disables simultaneous compilation for CUDA and HIP devices.")
      endif()
    endif()
  endif()
  if(${LLVM_VERSION_MAJOR} LESS 11 AND ${WITH_ACCELERATED_CPU})
    message(WARNING "clang version too old (${LLVM_VERSION_MAJOR} < 11) to be used with CPU acceleration support, disabling WITH_ACCELERATED_CPU")
    set(WITH_ACCELERATED_CPU false)
  endif()
  if(${LLVM_VERSION_MAJOR} LESS 14 AND ${WITH_SSCP_COMPILER})
    message(WARNING "clang version too old (${LLVM_VERSION_MAJOR} < 14) to be used with SSCP compiler, disabling WITH_SSCP_COMPILER")
    set(WITH_SSCP_COMPILER false)
  endif()
  if(${LLVM_VERSION_MAJOR} LESS 14 AND ${WITH_STDPAR_COMPILER})
    message(WARNING "clang version too old (${LLVM_VERSION_MAJOR} < 14) to be used with C++ standard parallelism support, disabling WITH_STDPAR_COMPILER")
    set(WITH_STDPAR_COMPILER false)
  endif()
  message(STATUS "Using clang include directory: ${CLANG_INCLUDE_PATH}")
endif()

#add_compile_definitions(HIPSYCL_DEBUG_LEVEL="${HIPSYCL_DEBUG_LEVEL}")
#Use add_definitions for now for older cmake versions
cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})
option(HIPSYCL_NO_DEVICE_MANGLER "Disable the use of DeviceMangler. necessary for LLVM 13 distributed with the rocm packages." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src)

set(SYCLCC_CONFIG_FILE_PATH "${PROJECT_BINARY_DIR}/syclcc.json")
set(SYCLCC_CONFIG_FILE_GLOBAL_INSTALLATION false CACHE BOOL 
  "Whether to install the syclcc configuration file into a global directory (typically, /etc/hipSYCL). This is generally not recommended.")

if(SYCLCC_CONFIG_FILE_GLOBAL_INSTALLATION)
  set(SYCLCC_CONFIG_FILE_INSTALL_DIR /etc/hipSYCL/)
else()
  set(SYCLCC_CONFIG_FILE_INSTALL_DIR etc/hipSYCL)
endif()

if(Boost_FIBER_LIBRARY_DEBUG)
  get_filename_component(Boost_LIBRARY_DIR "${Boost_FIBER_LIBRARY_DEBUG}" DIRECTORY)
elseif(Boost_FIBER_LIBRARY_RELEASE)
  get_filename_component(Boost_LIBRARY_DIR "${Boost_FIBER_LIBRARY_RELEASE}" DIRECTORY)
else()
  get_target_property(FIBER_IMPORT_LIBRARY Boost::fiber IMPORTED_LOCATION_DEBUG)
  if(NOT FIBER_IMPORT_LIBRARY)
    get_target_property(FIBER_IMPORT_LIBRARY Boost::fiber IMPORTED_LOCATION_RELEASE)
  endif()
  get_filename_component(Boost_LIBRARY_DIR "${FIBER_IMPORT_LIBRARY}" DIRECTORY)
  get_target_property(Boost_INCLUDE_DIR Boost::fiber INTERFACE_INCLUDE_DIRECTORIES)
endif()

if(APPLE)
  set(DEFAULT_OMP_FLAG "-Xclang -fopenmp")
  
  if(Boost_FIBER_LIBRARY_DEBUG)
    set(DEFAULT_BOOST_LIBRARIES "${Boost_CONTEXT_LIBRARY_DEBUG} ${Boost_FIBER_LIBRARY_DEBUG} -Wl,-rpath ${Boost_LIBRARY_DIR}")
  else()
    set(DEFAULT_BOOST_LIBRARIES "${Boost_CONTEXT_LIBRARY_RELEASE} ${Boost_FIBER_LIBRARY_RELEASE} -Wl,-rpath ${Boost_LIBRARY_DIR}")
  endif()
else()
  set(DEFAULT_OMP_FLAG "-fopenmp")
endif()

set(DEFAULT_GPU_ARCH "" CACHE STRING "Optional: Default GPU architecture to compile for when targeting GPUs (e.g.: sm_60 or gfx900)")
set(ROCM_LIBS "-L${ROCM_PATH}/lib -L${ROCM_PATH}/hip/lib -lamdhip64" CACHE STRING "Necessary libraries for ROCm")

set(DEFAULT_WIN32_CUDA_LINK_LINE "-L$HIPSYCL_CUDA_LIB_PATH -lcudart")
set(DEFAULT_CUDA_LINK_LINE "-Wl,-rpath=$HIPSYCL_CUDA_LIB_PATH -L$HIPSYCL_CUDA_LIB_PATH -lcudart")

if(MINGW)
  set(DEFAULT_LINK_PREFIX "-l:")
else()
  set(DEFAULT_LINK_PREFIX "-l")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  get_filename_component(Boost_CONTEXT_LIBRARY_NAME "${Boost_CONTEXT_LIBRARY_DEBUG}" NAME)
  get_filename_component(Boost_FIBER_LIBRARY_NAME "${Boost_FIBER_LIBRARY_DEBUG}" NAME)
else()
  get_filename_component(Boost_CONTEXT_LIBRARY_NAME "${Boost_CONTEXT_LIBRARY_RELEASE}" NAME)
  get_filename_component(Boost_FIBER_LIBRARY_NAME "${Boost_FIBER_LIBRARY_RELEASE}" NAME)
endif()
set(DEFAULT_WIN32_OMP_LINK_LINE "-L${Boost_LIBRARY_DIR} ${DEFAULT_LINK_PREFIX}${Boost_CONTEXT_LIBRARY_NAME} ${DEFAULT_LINK_PREFIX}${Boost_FIBER_LIBRARY_NAME} ${DEFAULT_OMP_FLAG}")
set(DEFAULT_WIN32_SEQUENTIAL_LINK_LINE "-L${Boost_LIBRARY_DIR} ${DEFAULT_LINK_PREFIX}${Boost_CONTEXT_LIBRARY_NAME} ${DEFAULT_LINK_PREFIX}${Boost_FIBER_LIBRARY_NAME} -llibomp")

# need add_subdirectory(src) before this!
set(DEFAULT_APPLE_OMP_LINK_LINE "${DEFAULT_BOOST_LIBRARIES} ${DEFAULT_OMP_FLAG} ${hipSYCL_OpenMP_CXX_LIBRARIES}")
set(DEFAULT_APPLE_SEQUENTIAL_LINK_LINE "${DEFAULT_BOOST_LIBRARIES} ${hipSYCL_OpenMP_CXX_LIBRARIES}")
set(DEFAULT_OMP_LINK_LINE "-L${Boost_LIBRARY_DIR} -lboost_context -lboost_fiber -Wl,-rpath=${Boost_LIBRARY_DIR} ${DEFAULT_OMP_FLAG}")
set(DEFAULT_SEQUENTIAL_LINK_LINE "-L${Boost_LIBRARY_DIR} -lboost_context -lboost_fiber -Wl,-rpath=${Boost_LIBRARY_DIR}")

# If no link lines given, set to default.
if(NOT ROCM_LINK_LINE)
  set(ROCM_LINK_LINE "-Wl,-rpath=$HIPSYCL_ROCM_PATH/lib -Wl,-rpath=$HIPSYCL_ROCM_PATH/hip/lib ${ROCM_LIBS}" CACHE STRING "Arguments passed to compiler to link ROCm libraries to SYCL applications")
endif()

if(WIN32)
  if(NOT CUDA_LINK_LINE)
    set(CUDA_LINK_LINE ${DEFAULT_WIN32_CUDA_LINK_LINE} CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
  endif()
  if(NOT OMP_LINK_LINE)
    set(OMP_LINK_LINE ${DEFAULT_WIN32_OMP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link OpenMP libraries to SYCL applications")
  endif()
  if(NOT SEQUENTIAL_LINK_LINE) 
    set(SEQUENTIAL_LINK_LINE ${DEFAULT_WIN32_SEQUENTIAL_LINK_LINE} CACHE STRING "Arguments passed to compiler to link host libraries to SYCL applications")
  endif()
elseif(APPLE)
  if(NOT CUDA_LINK_LINE) # no CUDA support here.. but make CMake happy.
    set(CUDA_LINK_LINE "" CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
  endif()
  if(NOT OMP_LINK_LINE)
    set(OMP_LINK_LINE ${DEFAULT_APPLE_OMP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link OpenMP libraries to SYCL applications")
  endif()
  if(NOT SEQUENTIAL_LINK_LINE) 
    set(SEQUENTIAL_LINK_LINE ${DEFAULT_APPLE_SEQUENTIAL_LINK_LINE} CACHE STRING "Arguments passed to compiler to link host libraries to SYCL applications")
  endif()
else()
  if(NOT CUDA_LINK_LINE)
    set(CUDA_LINK_LINE ${DEFAULT_CUDA_LINK_LINE} CACHE STRING "Arguments passed to compiler to link CUDA libraries to SYCL applications")
  endif()
  if(NOT OMP_LINK_LINE)
    set(OMP_LINK_LINE ${DEFAULT_OMP_LINK_LINE} CACHE STRING "Arguments passed to compiler to link OpenMP libraries to SYCL applications")
  endif()
  if(NOT SEQUENTIAL_LINK_LINE) 
    set(SEQUENTIAL_LINK_LINE ${DEFAULT_SEQUENTIAL_LINK_LINE} CACHE STRING "Arguments passed to compiler to link host libraries to SYCL applications")
  endif()
endif()

# If no compile flags given, set to default.
if(NOT ROCM_CXX_FLAGS)
  # clang erroneously sets feature detection flags for 
  # __float128 even though it is not supported for CUDA / HIP,
  # see https://bugs.llvm.org/show_bug.cgi?id=47559.

  set(ROCM_CXX_FLAGS "-isystem $HIPSYCL_PATH/include/hipSYCL/std/hiplike -isystem ${CLANG_INCLUDE_PATH} -U__FLOAT128__ -U__SIZEOF_FLOAT128__ -I$HIPSYCL_ROCM_PATH/include -I$HIPSYCL_ROCM_PATH/include --rocm-device-lib-path=$HIPSYCL_ROCM_PATH/amdgcn/bitcode --rocm-path=$HIPSYCL_ROCM_PATH -fhip-new-launch-api -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -D__HIP_ROCclr__" CACHE STRING "Arguments passed to compiler to compile SYCL applications with ROCm")
endif()

if(NOT CUDA_CXX_FLAGS)	
  # clang erroneously sets feature detection flags for 
  # __float128 even though it is not supported for CUDA / HIP,
  # see https://bugs.llvm.org/show_bug.cgi?id=47559.
  set(CUDA_CXX_FLAGS "-U__FLOAT128__ -U__SIZEOF_FLOAT128__ -isystem $HIPSYCL_PATH/include/hipSYCL/std/hiplike" CACHE STRING "Arguments passed to compiler to compile SYCL applications with CUDA")
endif()

# always need -D_ENABLE_EXTENDED_ALIGNED_STORAGE to allow correctly aligned local memory on CPU
if(NOT OMP_CXX_FLAGS) 
  set(OMP_CXX_FLAGS "-I${Boost_INCLUDE_DIR} ${DEFAULT_OMP_FLAG} -D_ENABLE_EXTENDED_ALIGNED_STORAGE" CACHE STRING "Arguments passed to compiler to compile SYCL applications with OpenMP")
endif()

if(NOT SEQUENTIAL_CXX_FLAGS) 
  set(SEQUENTIAL_CXX_FLAGS "-I${Boost_INCLUDE_DIR} -D_ENABLE_EXTENDED_ALIGNED_STORAGE" CACHE STRING "Arguments passed to compiler to compile SYCL applications on host")
endif()

if(BUILD_CLANG_PLUGIN)
  set(PLUGIN_LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
else()
  set(PLUGIN_LLVM_VERSION_MAJOR 0)
endif()

set(SYCLCC_CONFIG_FILE "{
  \"version-major\" : \"${HIPSYCL_VERSION_MAJOR}\",
  \"version-minor\" : \"${HIPSYCL_VERSION_MINOR}\",
  \"version-patch\" : \"${HIPSYCL_VERSION_PATCH}\",
  \"plugin-llvm-version-major\" : \"${PLUGIN_LLVM_VERSION_MAJOR}\",
  \"plugin-with-cpu-acceleration\" : \"${WITH_ACCELERATED_CPU}\",
  \"default-clang\"     : \"${CLANG_EXECUTABLE_PATH}\",
  \"default-nvcxx\"     : \"${NVCXX_COMPILER}\",
  \"default-platform\"  : \"${DEFAULT_PLATFORM}\",
  \"default-cuda-path\" : \"${CUDA_TOOLKIT_ROOT_DIR}\",
  \"default-gpu-arch\"  : \"${DEFAULT_GPU_ARCH}\",
  \"default-cpu-cxx\"   : \"${CMAKE_CXX_COMPILER}\",
  \"default-rocm-path\" : \"${ROCM_PATH}\",
  \"default-use-bootstrap-mode\" : \"false\",
  \"default-is-dryrun\" : \"false\",
  \"default-use-accelerated-cpu\" : \"${WITH_ACCELERATED_CPU}\",
  \"default-clang-include-path\" : \"${CLANG_INCLUDE_PATH}\",
  \"default-sequential-link-line\" : \"${SEQUENTIAL_LINK_LINE}\",
  \"default-sequential-cxx-flags\" : \"${SEQUENTIAL_CXX_FLAGS}\",
  \"default-omp-link-line\" : \"${OMP_LINK_LINE}\",
  \"default-omp-cxx-flags\" : \"${OMP_CXX_FLAGS}\",
  \"default-rocm-link-line\" : \"${ROCM_LINK_LINE}\",
  \"default-rocm-cxx-flags\" : \"${ROCM_CXX_FLAGS}\",
  \"default-cuda-link-line\" : \"${CUDA_LINK_LINE}\",
  \"default-cuda-cxx-flags\" : \"${CUDA_CXX_FLAGS}\",
  \"default-is-explicit-multipass\" : \"false\",
  \"default-save-temps\" : \"false\"
}
")

file(WRITE ${SYCLCC_CONFIG_FILE_PATH} ${SYCLCC_CONFIG_FILE})



# must stay after add_subdirectory(src) as HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME is declared there
configure_file(
  ${PROJECT_SOURCE_DIR}/include/hipSYCL/common/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/hipSYCL/common/config.hpp)

install(DIRECTORY include/CL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/SYCL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/hipSYCL DESTINATION include/ FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY include/hipSYCL DESTINATION include/ FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/hipSYCL/std DESTINATION include/hipSYCL/ )
install(FILES ${PROJECT_BINARY_DIR}/include/hipSYCL/common/config.hpp DESTINATION include/hipSYCL/common/)

if(NOT WIN32)
# Windows is case-insensitive, so don't copy to sycl/sycl.hpp as
# we already have SYCL/sycl.hpp
install(FILES include/SYCL/sycl.hpp DESTINATION include/sycl/)
endif()
install(PROGRAMS bin/acpp DESTINATION bin)
install(PROGRAMS bin/acpp DESTINATION bin RENAME syclcc)
install(PROGRAMS bin/acpp DESTINATION bin RENAME syclcc-clang)


install(FILES ${SYCLCC_CONFIG_FILE_PATH} DESTINATION ${SYCLCC_CONFIG_FILE_INSTALL_DIR})

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)



# Legacy module (hipSYCL)


set(HIPSYCL_INSTALL_CMAKE_DIR
  "lib/cmake/hipSYCL" CACHE PATH "Install path for legacy CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is
# the future install root
set(HIPSYCL_INSTALL_COMPILER_DIR bin)
set(HIPSYCL_INSTALL_LAUNCHER_DIR ${HIPSYCL_INSTALL_CMAKE_DIR})
set(HIPSYCL_INSTALL_LAUNCHER_RULE_DIR ${HIPSYCL_INSTALL_CMAKE_DIR})

install(PROGRAMS cmake/syclcc-launcher DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR})
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR})

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/hipsycl-config-version.cmake"
  VERSION ${hipSYCL_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/hipsycl-config.cmake.in
    ${PROJECT_BINARY_DIR}/hipsycl-config.cmake
    INSTALL_DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR}
    PATH_VARS
    HIPSYCL_INSTALL_COMPILER_DIR
    HIPSYCL_INSTALL_LAUNCHER_DIR
    HIPSYCL_INSTALL_LAUNCHER_RULE_DIR
)

install(FILES
  ${PROJECT_BINARY_DIR}/hipsycl-config.cmake
  ${PROJECT_BINARY_DIR}/hipsycl-config-version.cmake
  DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR}
)

install(EXPORT install_exports
  FILE "hipsycl-targets.cmake"
  NAMESPACE hipSYCL::
  DESTINATION ${HIPSYCL_INSTALL_CMAKE_DIR}
)

mark_as_advanced(
  HIPSYCL_INSTALL_CMAKE_DIR
)


# Legacy module (Open SYCL)

set(OPENSYCL_INSTALL_CMAKE_DIR
  "lib/cmake/OpenSYCL" CACHE PATH "Install path for CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is
# the future install root
set(OPENSYCL_INSTALL_COMPILER_DIR bin)
set(OPENSYCL_INSTALL_LAUNCHER_DIR ${OPENSYCL_INSTALL_CMAKE_DIR})
set(OPENSYCL_INSTALL_LAUNCHER_RULE_DIR ${OPENSYCL_INSTALL_CMAKE_DIR})

install(PROGRAMS cmake/syclcc-launcher DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR})
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR})

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/opensycl-config-version.cmake"
  VERSION ${hipSYCL_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/opensycl-config.cmake.in
    ${PROJECT_BINARY_DIR}/opensycl-config.cmake
    INSTALL_DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR}
    PATH_VARS
    OPENSYCL_INSTALL_COMPILER_DIR
    OPENSYCL_INSTALL_LAUNCHER_DIR
    OPENSYCL_INSTALL_LAUNCHER_RULE_DIR
)

install(FILES
  ${PROJECT_BINARY_DIR}/opensycl-config.cmake
  ${PROJECT_BINARY_DIR}/opensycl-config-version.cmake
  DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR}
)

install(EXPORT install_exports
  FILE "opensycl-targets.cmake"
  NAMESPACE OpenSYCL::
  DESTINATION ${OPENSYCL_INSTALL_CMAKE_DIR}
)

mark_as_advanced(
  OPENSYCL_INSTALL_CMAKE_DIR
)

# New module


set(ADAPTIVECPP_INSTALL_CMAKE_DIR
  "lib/cmake/AdaptiveCpp" CACHE PATH "Install path for CMake config files")

# Set relative paths for install root in the following variables so that
# configure_package_config_file will generate paths relative whatever is
# the future install root
set(ADAPTIVECPP_INSTALL_COMPILER_DIR bin)
set(ADAPTIVECPP_INSTALL_LAUNCHER_DIR ${ADAPTIVECPP_INSTALL_CMAKE_DIR})
set(ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR ${ADAPTIVECPP_INSTALL_CMAKE_DIR})

install(PROGRAMS cmake/syclcc-launcher DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR})
install(PROGRAMS cmake/syclcc-launch.rule.in DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR})

write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/adaptivecpp-config-version.cmake"
  VERSION ${hipSYCL_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/adaptivecpp-config.cmake.in
    ${PROJECT_BINARY_DIR}/adaptivecpp-config.cmake
    INSTALL_DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
    PATH_VARS
    ADAPTIVECPP_INSTALL_COMPILER_DIR
    ADAPTIVECPP_INSTALL_LAUNCHER_DIR
    ADAPTIVECPP_INSTALL_LAUNCHER_RULE_DIR
)

install(FILES
  ${PROJECT_BINARY_DIR}/adaptivecpp-config.cmake
  ${PROJECT_BINARY_DIR}/adaptivecpp-config-version.cmake
  DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
)

install(EXPORT install_exports
  FILE "adaptivecpp-targets.cmake"
  NAMESPACE AdaptiveCpp::
  DESTINATION ${ADAPTIVECPP_INSTALL_CMAKE_DIR}
)

mark_as_advanced(
  ADAPTIVECPP_INSTALL_CMAKE_DIR
)

