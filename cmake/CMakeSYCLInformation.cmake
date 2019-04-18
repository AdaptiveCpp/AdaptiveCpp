# ---------------------
# hipSYCL-specific setup
# ---------------------

set(HIPSCL_NUM_AVAILABLE_BACKENDS 0)
if(HIPSYCL_CPU_BACKEND_AVAILABLE)
  MATH(EXPR HIPSYCL_NUM_AVAILABLE_BACKENDS "${HIPSYCL_NUM_AVAILABLE_BACKENDS}+1")
  if(NOT HIPSYCL_DEFAULT_PLATFORM)
    set(HIPSYCL_DEFAULT_PLATFORM "cpu")
  endif()
endif()
if(HIPSYCL_CUDA_BACKEND_AVAILABLE)
  MATH(EXPR HIPSYCL_NUM_AVAILABLE_BACKENDS "${HIPSYCL_NUM_AVAILABLE_BACKENDS}+1")
  if(NOT HIPSYCL_DEFAULT_PLATFORM)
    set(HIPSYCL_DEFAULT_PLATFORM "cuda")
  endif()
endif()
if(HIPSYCL_ROCM_BACKEND_AVAILABLE)
  MATH(EXPR HIPSYCL_NUM_AVAILABLE_BACKENDS "${HIPSYCL_NUM_AVAILABLE_BACKENDS}+1")
  if(NOT HIPSYCL_DEFAULT_PLATFORM)
    set(HIPSYCL_DEFAULT_PLATFORM "rocm")
  endif()
endif()

set(HIPSYCL_PLATFORM "" CACHE STRING "The hipSYCL backend to use. One of cpu|cuda|nvcc|rocm.")

# If HIPSYCL_PLATFORM has not been explicitly set by the user, first try to find
# the corresponding environment variable. If that isn't set either, and only
# a single platform is available, default to it. Otherwise throw an error.
if(NOT HIPSYCL_PLATFORM)
  set(HIPSYCL_PLATFORM_ENV $ENV{HIPSYCL_PLATFORM})
  if(HIPSYCL_PLATFORM_ENV)
    message("Found HIPSYCL_PLATFORM from environment: ${HIPSYCL_PLATFORM_ENV}")
    set(HIPSYCL_DEFAULT_PLATFORM ${HIPSYCL_PLATFORM_ENV})
  elseif(HIPSYCL_NUM_AVAILABLE_BACKENDS GREATER 1)
    message(SEND_ERROR "More than one hipSYCL backend is available.\n"
            "Please specify HIPSYCL_PLATFORM=cpu|cuda|nvcc|rocm")
  endif()
  set(HIPSYCL_PLATFORM ${HIPSYCL_DEFAULT_PLATFORM})
  unset(HIPSYCL_PLATFORM_ENV)
endif()

unset(HIPSYCL_NUM_AVAILABLE_BACKENDS)
unset(HIPSYCL_DEFAULT_PLATFORM)

set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --hipsycl-platform=${HIPSYCL_PLATFORM}")

# Less critical options

# We have to handle this one a bit differently, as we do want to be able to set it
# to the empty string (which means that no headers should be rewritten).
if(NOT DEFINED HIPSYCL_RESTRICT_DEVICE_HEADER_PATH)
  set(HIPSYCL_RDHP_INIT "<no-restriction>")
else()
  set(HIPSYCL_RDHP_INIT ${HIPSYCL_RESTRICT_DEVICE_HEADER_PATH})
endif()
set(HIPSYCL_RESTRICT_DEVICE_HEADER_PATH "${HIPSYCL_RDHP_INIT}" CACHE STRING "List of paths of where headers containing device code can be found.")
unset(HIPSYCL_RDHP_INIT)
if(NOT HIPSYCL_RESTRICT_DEVICE_HEADER_PATH MATCHES "<no-restriction>")
  if(NOT HIPSYCL_RESTRICT_DEVICE_HEADER_PATH)
    set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --restrict-device-header-path=\"\"")
  endif()
  foreach(path ${HIPSYCL_RESTRICT_DEVICE_HEADER_PATH})
    set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --restrict-device-header-path=\"${path}\"")
  endforeach()
endif()

set(HIPSYCL_CUDA_CLANG_COMPILER "" CACHE STRING "Clang compiler executable used for CUDA compilation.")
if(HIPSYCL_CUDA_CLANG_COMPILER)
  set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --cuda-clang-compiler=${HIPSYCL_CUDA_CLANG_COMPILER}")
endif()

set(HIPSYCL_KEEP_TEMPORARY_FILES FALSE CACHE BOOL "Whether to keep temporary files after the compilation has finished.")
if(HIPSYCL_KEEP_TEMPORARY_FILES)
  set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --keep-temporary-files")
endif()

set(HIPSYCL_GPU_ARCH "" CACHE STRING "GPU architecture used by ROCm / CUDA (when compiled with Clang).")
if(HIPSYCL_GPU_ARCH)
  # TODO: We could avoid this by adding an explicit parameter for syclcc instead of using the native ones
  if(HIPSYCL_PLATFORM MATCHES "cuda|nvidia")
    set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --cuda-gpu-arch=${HIPSYCL_GPU_ARCH}")
  elseif(HIPSYCL_PLATFORM MATCHES "rocm|amd|hip|hcc")
    set(CMAKE_SYCL_FLAGS_INIT "${CMAKE_SYCL_FLAGS_INIT} --amdgpu-target=${HIPSYCL_GPU_ARCH}")
  else()
    message(WARNING "HIPSYCL_GPU_ARCH (${HIPSYCL_GPU_ARCH}) is ignored for current backend (${HIPSYCL_PLATFORM})")
  endif()
endif()

# ---------------------
# Compilation flags and rules
# These are mostly based on https://github.com/Kitware/CMake/blob/master/Modules/CMakeCXXInformation.cmake
# CMake is distributed under the BSD 3-clause License.
# ---------------------

if(NOT CMAKE_INCLUDE_FLAG_SYCL)
  set(CMAKE_INCLUDE_FLAG_SYCL ${CMAKE_INCLUDE_FLAG_CXX})
endif()

set(CMAKE_SYCL_FLAGS_INIT "$ENV{SYCLFLAGS} ${CMAKE_SYCL_FLAGS_INIT}")
string(STRIP "${CMAKE_SYCL_FLAGS_INIT}" CMAKE_SYCL_FLAGS_INIT)

foreach(c _DEBUG _RELEASE _MINSIZEREL _RELWITHDEBINFO)
  string(STRIP "${CMAKE_CXX_FLAGS${c}_INIT}" CMAKE_SYCL_FLAGS${c}_INIT)
endforeach()

set(CMAKE_SYCL_FLAGS "${CMAKE_SYCL_FLAGS_INIT}" CACHE STRING
    "Flags used by the compiler during all build types.")

if(NOT CMAKE_NOT_USING_CONFIG_FLAGS)
  set(CMAKE_SYCL_FLAGS_DEBUG "${CMAKE_SYCL_FLAGS_DEBUG_INIT}" CACHE STRING
      "Flags used by the compiler during debug builds.")
  set(CMAKE_SYCL_FLAGS_MINSIZEREL "${CMAKE_SYCL_FLAGS_MINSIZEREL_INIT}" CACHE STRING
      "Flags used by the compiler during release builds for minimum size.")
  set(CMAKE_SYCL_FLAGS_RELEASE "${CMAKE_SYCL_FLAGS_RELEASE_INIT}" CACHE STRING
      "Flags used by the compiler during release builds.")
  set(CMAKE_SYCL_FLAGS_RELWITHDEBINFO "${CMAKE_SYCL_FLAGS_RELWITHDEBINFO_INIT}" CACHE STRING
      "Flags used by the compiler during release builds with debug info.")
endif()

include(CMakeCommonLanguageInclude)

if(NOT CMAKE_SYCL_CREATE_SHARED_LIBRARY)
  set(CMAKE_SYCL_CREATE_SHARED_LIBRARY
      "<CMAKE_SYCL_COMPILER> <CMAKE_SHARED_LIBRARY_CXX_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

if(NOT CMAKE_SYCL_CREATE_SHARED_MODULE)
  set(CMAKE_SYCL_CREATE_SHARED_MODULE ${CMAKE_SYCL_CREATE_SHARED_LIBRARY})
endif()

if(NOT DEFINED CMAKE_SYCL_ARCHIVE_CREATE)
  set(CMAKE_SYCL_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_SYCL_ARCHIVE_APPEND)
  set(CMAKE_SYCL_ARCHIVE_APPEND "<CMAKE_AR> q  <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_SYCL_ARCHIVE_FINISH)
  set(CMAKE_SYCL_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

if(NOT CMAKE_SYCL_COMPILE_OBJECT)
  set(CMAKE_SYCL_COMPILE_OBJECT "<CMAKE_SYCL_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_SYCL_LINK_EXECUTABLE)
  set(CMAKE_SYCL_LINK_EXECUTABLE "<CMAKE_SYCL_COMPILER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
endif()

set(CMAKE_SYCL_INFORMATION_LOADED 1)

