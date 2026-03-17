if(WITH_LLVM_TO_VULKAN)
  libkernel_generate_bitcode_target(
      TARGETNAME clspv
      TRIPLE spir64-unknown-unknown
      SOURCES
      assert.cpp
      broadcast.cpp
      print.cpp
      atomic.cpp
      collpredicate.cpp
      integer.cpp
      half.cpp
      relational.cpp
      math.cpp
      native.cpp
      barrier.cpp
      localmem.cpp
      subgroup.cpp
      reduction.cpp
      shuffle.cpp
      scan_exclusive.cpp
      scan_inclusive.cpp
      core.cpp)
endif()
