
if(COMPILE_SOURCE_TRANSFORMATION_TOOLS)
  add_subdirectory(hipsycl_rewrite_includes)
  add_subdirectory(hipsycl_transform_source)
ENDIF()
if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
endif()
add_subdirectory(runtime)
