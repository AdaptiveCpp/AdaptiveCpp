if(WITH_LLVM_TO_MUSA)
  libkernel_generate_bitcode_target(
      TARGETNAME mtgpu
      TRIPLE mtgpu-mt-musa
      SOURCES atomic.cpp barrier.cpp core.cpp half.cpp integer.cpp relational.cpp math.cpp native.cpp localmem.cpp subgroup.cpp
      ADDITIONAL_ARGS -D__MTGPU__ -Xclang -target-feature -Xclang +mp_10 -mllvm -mtgpu-internalize-symbols=false)
endif()
