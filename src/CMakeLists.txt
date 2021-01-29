if(WIN32)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS On)
endif()

if(BUILD_CLANG_PLUGIN)
  add_subdirectory(compiler)
endif()
add_subdirectory(runtime)
