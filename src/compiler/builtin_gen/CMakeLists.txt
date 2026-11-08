add_executable(acpp-amdgpu-builtin-gen main.cpp)

configure_target(TARGET acpp-amdgpu-builtin-gen)

# Unconditionally disable RTTI and Exceptions for this tool.
# We don't use them, and if Clang/LLVM were built without them, failing to pass
# these flags will result in undefined references to typeinfo for ASTFrontendAction.
target_compile_options(acpp-amdgpu-builtin-gen PRIVATE -fno-exceptions -fno-rtti)

if(NOT WIN32 AND NOT ACPP_LLVM_COMPONENT)
  # On Unix, we need to explicitly link the clang-cpp and LLVM shared libraries
  # since configure_target is tailored for plugins and doesn't link them automatically.
  find_library(CLANG_CPP_LIB NAMES clang-cpp PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
  find_library(LLVM_LIB NAMES LLVM PATHS ${LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)

  if (NOT CLANG_CPP_LIB)
    # If clang-cpp dynamic library is missing, try to find Clang CMake package for static targets
    find_package(Clang QUIET HINTS "${LLVM_LIBRARY_DIR}/cmake/clang" "${LLVM_DIR}/../clang")
    if (TARGET clangTooling)
      set(CLANG_CPP_LIB clangTooling)
    else()
      # Manual fallback to static libraries with start/end group for circular deps
      set(CLANG_CPP_LIB
        -Wl,--start-group
        clangTooling clangFrontend clangParse clangSerialization clangSema 
        clangEdit clangAnalysis clangASTMatchers clangAST clangLex clangBasic clangDriver
        -Wl,--end-group
      )
      target_link_directories(acpp-amdgpu-builtin-gen PRIVATE ${LLVM_LIBRARY_DIR})
    endif()
  endif()
  
  if (NOT LLVM_LIB)
    set(LLVM_LIB LLVM)
  endif()

  target_link_options(acpp-amdgpu-builtin-gen PRIVATE "-Wl,-rpath,${LLVM_LIBRARY_DIR}")

  target_link_libraries(acpp-amdgpu-builtin-gen PRIVATE
    ${CLANG_CPP_LIB}
    ${LLVM_LIB}
  )
endif()

if(ACPP_LLVM_COMPONENT)
  # When building in-tree with LLVM, the Clang targets are directly available.
  target_link_libraries(acpp-amdgpu-builtin-gen PRIVATE
    clangTooling clangFrontend clangParse clangSerialization clangSema
    clangEdit clangAnalysis clangASTMatchers clangAST clangLex clangBasic clangDriver
  )

  # We need to explicitly include the Clang headers
  # because LLVM_INCLUDE_DIRS only covers LLVM, not Clang.
  target_include_directories(acpp-amdgpu-builtin-gen PRIVATE
    ${LLVM_MAIN_SRC_DIR}/../clang/include
    ${LLVM_BINARY_DIR}/tools/clang/include
  )
endif()

# Optional: ensure we compile with C++17 or whatever AdaptiveCpp uses
set_property(TARGET acpp-amdgpu-builtin-gen PROPERTY CXX_STANDARD 17)

