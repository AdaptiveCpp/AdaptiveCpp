
if(WITH_LLVM_TO_SPIRV)
  libkernel_generate_bitcode_target(
      TARGETNAME spirv 
      TRIPLE spir64-unknown-unknown
      SOURCES barrier.cpp core.cpp math.cpp native.cpp integer.cpp relational.cpp localmem.cpp subgroup.cpp)
endif()