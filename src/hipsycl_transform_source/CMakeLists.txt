project(hipsycl_transform_source)
cmake_minimum_required(VERSION 3.5)

find_package(LLVM REQUIRED)
#This seems to configure linkage incorrectly, so let's just assume clang is installed
#find_package(Clang REQUIRED)
find_package(Boost REQUIRED)

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

#target_compile_options(hipsycl_transform_source PRIVATE
#  --force-alternative-compiler=g++)
#set_target_properties(hipsycl_transform_source PROPERTIES LINK_FLAGS
#  --force-alternative-compiler=gcc)

link_directories(${LLVM_LIBRARY_DIRS})

target_compile_definitions(hipsycl_transform_source PRIVATE
  ${LLVM_DEFINITIONS}
  HIPSYCL_TRANSFORM_CLANG_DIR=${CLANG_BINARY_PREFIX})

target_link_libraries(hipsycl_transform_source
  -lLLVM
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
