if(WITH_LLVM_TO_VULKAN)
  libkernel_generate_bitcode_target(
      TARGETNAME clspv
      TRIPLE spir64-unknown-unknown
      SOURCES
      assert.cpp
      print.cpp
      atomic.cpp
      integer.cpp
      half.cpp
      relational.cpp
      math.cpp
      native.cpp
      barrier.cpp
      localmem.cpp
      subgroup.cpp
      core.cpp)
endif()
