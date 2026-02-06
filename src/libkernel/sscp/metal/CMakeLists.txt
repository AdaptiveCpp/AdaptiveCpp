if(WITH_LLVM_TO_METAL)
  set(METAL_LIBKERNEL_BITCODE_SOURCES
    math.cpp)

  libkernel_generate_bitcode_target(
      TARGETNAME metal
      TRIPLE "arm64-apple-macosx"
      SOURCES ${METAL_LIBKERNEL_BITCODE_SOURCES})
endif()
