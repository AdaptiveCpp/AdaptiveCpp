if(WITH_LLVM_TO_AMDGPU_AMDHSA)
  libkernel_generate_bitcode_target(
      TARGETNAME amdgpu-amdhsa 
      TRIPLE amdgcn-amd-amdhsa
      SOURCES 
      assert.cpp
      atomic.cpp 
      barrier.cpp 
      core.cpp
      half.cpp
      integer.cpp
      math.cpp
      native.cpp
      print.cpp
      relational.cpp
      subgroup.cpp
      localmem.cpp
      shuffle.cpp
      collpredicate.cpp
      reduction.cpp
      broadcast.cpp
      scan_inclusive.cpp
      scan_exclusive.cpp
      general_builtins.cpp 
      ADDITIONAL_ARGS -nogpulib)
    
    #TODO These will generate bc files with the full prefix as well.
    #Probably should not do that for cleanliness
    #This should be also changed in LLVMToAmdgpu.cpp as well
    libkernel_generate_bitcode_target(
        TARGETNAME amdgpu-amdhsa-atomic-gfx90a 
        TRIPLE amdgcn-amd-amdhsa
        SOURCES 
        atomic_gfx90a.cpp 
        ADDITIONAL_ARGS -nogpulib -mcpu=gfx90a)

    libkernel_generate_bitcode_target(
        TARGETNAME amdgpu-amdhsa-atomic-gfx90a_fallback
        TRIPLE amdgcn-amd-amdhsa
        SOURCES 
        atomic_gfx90a_fallback.cpp 
        ADDITIONAL_ARGS -nogpulib)

    # There seems to be no way to specifiy target-features for clang
    # So we just select an arch that has dpp instructions
    libkernel_generate_bitcode_target(
        TARGETNAME amdgpu-amdhsa-dpp_builtins 
        TRIPLE amdgcn-amd-amdhsa
        SOURCES 
        dpp_builtins.cpp 
        ADDITIONAL_ARGS -nogpulib -mcpu=gfx90c)

endif()
