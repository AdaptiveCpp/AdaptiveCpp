
find_package(LLVM REQUIRED)
#This seems to configure linkage incorrectly, so let's just assume clang is installed
#find_package(Clang REQUIRED)
find_package(Boost REQUIRED)

add_executable(hipsycl_rewrite_includes
  HipsyclRewriteIncludes.cpp
  InclusionRewriter.cpp)

target_include_directories(hipsycl_rewrite_includes PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS})

#target_compile_options(hipsycl_transform_source PRIVATE
#  --force-alternative-compiler=g++)
#set_target_properties(hipsycl_transform_source PROPERTIES LINK_FLAGS
#  --force-alternative-compiler=gcc)

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
