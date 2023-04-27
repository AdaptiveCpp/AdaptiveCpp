if(WITH_LLVM_TO_MUSA)
  libkernel_generate_bitcode_target(
      TARGETNAME mtgpu
      TRIPLE mtgpu-mt-cuda
      SOURCES atomic.cpp barrier.cpp core.cpp half.cpp integer.cpp relational.cpp math.cpp native.cpp localmem.cpp subgroup.cpp
      ADDITIONAL_ARGS -Xclang -target-feature -Xclang +sm_60 -Xclang -target-feature -Xclang +ptx60)
endif()
