#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "AdaptiveCpp::acpp-common" for configuration "Debug"
set_property(TARGET AdaptiveCpp::acpp-common APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(AdaptiveCpp::acpp-common PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libacpp-common.so"
  IMPORTED_SONAME_DEBUG "libacpp-common.so"
  )

list(APPEND _cmake_import_check_targets AdaptiveCpp::acpp-common )
list(APPEND _cmake_import_check_files_for_AdaptiveCpp::acpp-common "${_IMPORT_PREFIX}/lib/libacpp-common.so" )

# Import target "AdaptiveCpp::acpp-rt" for configuration "Debug"
set_property(TARGET AdaptiveCpp::acpp-rt APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(AdaptiveCpp::acpp-rt PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libacpp-rt.so"
  IMPORTED_SONAME_DEBUG "libacpp-rt.so"
  )

list(APPEND _cmake_import_check_targets AdaptiveCpp::acpp-rt )
list(APPEND _cmake_import_check_files_for_AdaptiveCpp::acpp-rt "${_IMPORT_PREFIX}/lib/libacpp-rt.so" )

# Import target "AdaptiveCpp::sycl_tracer" for configuration "Debug"
set_property(TARGET AdaptiveCpp::sycl_tracer APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(AdaptiveCpp::sycl_tracer PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libsycl_tracer.so"
  IMPORTED_SONAME_DEBUG "libsycl_tracer.so"
  )

list(APPEND _cmake_import_check_targets AdaptiveCpp::sycl_tracer )
list(APPEND _cmake_import_check_files_for_AdaptiveCpp::sycl_tracer "${_IMPORT_PREFIX}/lib/libsycl_tracer.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
