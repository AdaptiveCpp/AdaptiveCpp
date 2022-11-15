
add_library(hipSYCL-common SHARED
    filesystem.cpp)

target_include_directories(hipSYCL-common
  PUBLIC
    $<BUILD_INTERFACE:${HIPSYCL_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${HIPSYCL_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/include
)

list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake/")
find_package(Filesystem REQUIRED Final Experimental Boost)
set(CXX_FILESYSTEM_HEADER "${CXX_FILESYSTEM_HEADER}" PARENT_SCOPE)
set(CXX_FILESYSTEM_NAMESPACE "${CXX_FILESYSTEM_NAMESPACE}" PARENT_SCOPE)
list(REMOVE_AT CMAKE_MODULE_PATH 0)


# to get the search path we need to know the output name, so set it explicitly.
set(HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME "hipSYCL-common")
if(WIN32)
  set_target_properties(hipSYCL-common PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS On)

  # ensure actual output and expected are equivalent
  set_target_properties(hipSYCL-common PROPERTIES RUNTIME_OUTPUT_NAME ${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME})
  if(MinGW)
    set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME "${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.dll" PARENT_SCOPE)
  else()
    set(HIPSYCL_RT_LIBRARY_OUTPUT_NAME "lib${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.dll" PARENT_SCOPE)
  endif()
else()
  target_link_libraries(hipSYCL-common PRIVATE dl std::filesystem)

  # ensure actual output and expected are equivalent
  set_target_properties(hipSYCL-common PROPERTIES LIBRARY_OUTPUT_NAME ${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME})
  if(APPLE)
    set(HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME "lib${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.dylib" PARENT_SCOPE)
  else()
    set(HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME "lib${HIPSYCL_COMMON_LIBRARY_OUTPUT_NAME}.so" PARENT_SCOPE)
  endif()
endif()

install(TARGETS hipSYCL-common
        EXPORT install_exports
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)