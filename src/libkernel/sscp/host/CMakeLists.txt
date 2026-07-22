if(WITH_LLVM_TO_HOST)
  set(HOST_LIBKERNEL_BITCODE_SOURCES
    assert.cpp
    atomic.cpp
    barrier.cpp
    core.cpp
    integer.cpp
    half.cpp
    math.cpp
    native.cpp
    shuffle.cpp
    print.cpp
    relational.cpp
    localmem.cpp
    subgroup.cpp
    reduction.cpp
    broadcast.cpp
    scan_inclusive.cpp
    scan_exclusive.cpp
    collpredicate.cpp)

  set(HOST_BUILTINS_SYSROOT "")
  set(HOST_BUILTINS_TRIPLE "")
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(HOST_BUILTINS_SYSROOT --sysroot=${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot)
    set(HOST_BUILTINS_TRIPLE ${ANDROID_LLVM_TRIPLE})
  endif()

  libkernel_generate_bitcode_target(
      TARGETNAME host
      TRIPLE ${HOST_BUILTINS_TRIPLE}
      SOURCES ${HOST_LIBKERNEL_BITCODE_SOURCES}
      ADDITIONAL_ARGS -fno-math-errno -ffp-contract=fast ${HOST_BUILTINS_SYSROOT})

  libkernel_generate_bitcode_target(
      TARGETNAME host-fast
      TRIPLE ${HOST_BUILTINS_TRIPLE}
      SOURCES ${HOST_LIBKERNEL_BITCODE_SOURCES}
      ADDITIONAL_ARGS -ffast-math ${HOST_BUILTINS_SYSROOT})
endif()
