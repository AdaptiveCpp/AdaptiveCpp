cmake_minimum_required(VERSION 3.5)
project(hipsycl_transform_source)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED)
find_package(Boost REQUIRED)

find_program(CLANG_EXECUTABLE_PATH NAMES ${Clang_EXECUTABLE} CACHE STRING)
if(CLANG_EXECUTABLE_PATH MATCHES "-NOTFOUND")
  message(SEND_ERROR "Could not find clang executable in PATH")
endif()
get_filename_component(CLANG_BINARY_PREFIX ${CLANG_EXECUTABLE_PATH} DIRECTORY)

if(NOT HIPSYCL_DEBUG_LEVEL)
  set(HIPSYCL_DEBUG_LEVEL 1 CACHE INTEGER)
endif()

add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

add_executable(hipsycl_transform_source
  HipsyclTransformSource.cpp
  CompilationTargetAnnotator.cpp
  HipsyclTransform.cpp
  Matcher.cpp
  CallGraph.cpp)

target_include_directories(hipsycl_transform_source PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ../../include)

target_compile_definitions(hipsycl_transform_source PRIVATE
  ${LLVM_DEFINITIONS}
  HIPSYCL_TRANSFORM_CLANG_DIR=${CLANG_BINARY_PREFIX})

target_link_libraries(hipsycl_transform_source
  LLVM
  clangFrontend
  clangSerialization
  clangDriver
  clangParse
  clangSema
  clangAnalysis
  clangAST
  clangBasic
  clangEdit
  clangLex
  clangTooling
  clangRewrite
  clangRewriteFrontend
  clangASTMatchers)

install(TARGETS hipsycl_transform_source
        RUNTIME DESTINATION bin/)
