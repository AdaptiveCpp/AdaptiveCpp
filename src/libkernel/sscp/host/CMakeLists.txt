if(WITH_LLVM_TO_HOST)
  libkernel_generate_bitcode_target(
      TARGETNAME host 
      TRIPLE ${LLVM_TARGET_TRIPLE}
      SOURCES atomic.cpp barrier.cpp core.cpp integer.cpp half.cpp math.cpp native.cpp print.cpp relational.cpp localmem.cpp subgroup.cpp)
endif()
