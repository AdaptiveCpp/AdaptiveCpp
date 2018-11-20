project(hipsycl_rewrite_includes)
cmake_minimum_required (VERSION 3.5)

find_package(LLVM REQUIRED)
#This seems to configure linkage incorrectly, so let's just assume clang is installed
#find_package(Clang REQUIRED)
find_package(Boost REQUIRED)

add_executable(hipsycl_rewrite_includes
  HipsyclRewriteIncludes.cpp
  InclusionRewriter.cpp)

if(NOT HIPSYCL_DEBUG_LEVEL)
  set(HIPSYCL_DEBUG_LEVEL 1 CACHE INTEGER)
endif()

add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

target_include_directories(hipsycl_rewrite_includes PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ../../include)

target_compile_definitions(hipsycl_rewrite_includes PRIVATE
  ${LLVM_DEFINITIONS}
  HIPSYCL_TRANSFORM_CLANG_DIR=${CLANG_BINARY_PREFIX})


link_directories(${LLVM_LIBRARY_DIRS})

target_compile_definitions(hipsycl_rewrite_includes PRIVATE
  ${LLVM_DEFINITIONS})
target_link_libraries(hipsycl_rewrite_includes
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

install(TARGETS hipsycl_rewrite_includes
        RUNTIME DESTINATION bin/)
