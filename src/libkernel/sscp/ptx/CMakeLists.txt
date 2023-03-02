
if(WITH_LLVM_TO_PTX)
  libkernel_generate_bitcode_target(
      TARGETNAME ptx 
      TRIPLE nvptx64-nvidia-cuda
      SOURCES barrier.cpp core.cpp half.cpp integer.cpp relational.cpp math.cpp native.cpp localmem.cpp subgroup.cpp)
endif()
