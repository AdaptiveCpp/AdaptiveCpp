
if(WITH_LLVM_TO_SPIRV)
  libkernel_generate_bitcode_target(
      TARGETNAME spirv 
      TRIPLE spir64-unknown-unknown
      SOURCES atomic.cpp barrier.cpp core.cpp half.cpp math.cpp native.cpp integer.cpp print.cpp relational.cpp localmem.cpp subgroup.cpp)
endif()
