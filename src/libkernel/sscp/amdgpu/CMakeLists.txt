if(WITH_LLVM_TO_AMDGPU_AMDHSA)
  libkernel_generate_bitcode_target(
      TARGETNAME amdgpu-amdhsa 
      TRIPLE amdgcn-amd-amdhsa
      SOURCES atomic.cpp barrier.cpp core.cpp half.cpp integer.cpp math.cpp native.cpp print.cpp relational.cpp subgroup.cpp localmem.cpp
      ADDITIONAL_ARGS -nogpulib)
endif()
