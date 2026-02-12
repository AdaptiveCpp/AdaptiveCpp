if(WITH_LLVM_TO_METAL)
  set(METAL_LIBKERNEL_BITCODE_SOURCES
    assert.cpp
    atomic.cpp
    barrier.cpp
    broadcast.cpp
    collpredicate.cpp
    core.cpp
    half.cpp
    integer.cpp
    localmem.cpp
    math.cpp
    native.cpp
    print.cpp
    reduction.cpp
    scan_inclusive.cpp
    scan_exclusive.cpp
    shuffle.cpp
    subgroup.cpp)

  libkernel_generate_bitcode_target(
      TARGETNAME metal
      TRIPLE "arm64-apple-macosx"
      SOURCES ${METAL_LIBKERNEL_BITCODE_SOURCES})
endif()
