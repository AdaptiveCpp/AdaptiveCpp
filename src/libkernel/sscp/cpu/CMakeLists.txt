if(WITH_LLVM_TO_CPU)
  libkernel_generate_bitcode_target(
      TARGETNAME cpu 
      TRIPLE x86_64-pc-linux-gnu # fixme... 
      SOURCES barrier.cpp core.cpp integer.cpp math.cpp native.cpp print.cpp relational.cpp localmem.cpp) #atomic & subgroup missing
endif()
