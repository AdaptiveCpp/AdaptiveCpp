#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenSYCL::acpp-common" for configuration "Release"
set_property(TARGET OpenSYCL::acpp-common APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenSYCL::acpp-common PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libacpp-common.so"
  IMPORTED_SONAME_RELEASE "libacpp-common.so"
  )

list(APPEND _cmake_import_check_targets OpenSYCL::acpp-common )
list(APPEND _cmake_import_check_files_for_OpenSYCL::acpp-common "${_IMPORT_PREFIX}/lib/libacpp-common.so" )

# Import target "OpenSYCL::acpp-rt" for configuration "Release"
set_property(TARGET OpenSYCL::acpp-rt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenSYCL::acpp-rt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libacpp-rt.so"
  IMPORTED_SONAME_RELEASE "libacpp-rt.so"
  )

list(APPEND _cmake_import_check_targets OpenSYCL::acpp-rt )
list(APPEND _cmake_import_check_files_for_OpenSYCL::acpp-rt "${_IMPORT_PREFIX}/lib/libacpp-rt.so" )

# Import target "OpenSYCL::sycl_tracer" for configuration "Release"
set_property(TARGET OpenSYCL::sycl_tracer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenSYCL::sycl_tracer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsycl_tracer.so"
  IMPORTED_SONAME_RELEASE "libsycl_tracer.so"
  )

list(APPEND _cmake_import_check_targets OpenSYCL::sycl_tracer )
list(APPEND _cmake_import_check_files_for_OpenSYCL::sycl_tracer "${_IMPORT_PREFIX}/lib/libsycl_tracer.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
