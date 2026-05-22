set(amdgpu_libkernel_additional_args -nogpulib)

if(WIN32)
  list(APPEND amdgpu_libkernel_additional_args
    -Xclang -fwchar-type=short)
endif()

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
      ADDITIONAL_ARGS ${amdgpu_libkernel_additional_args})
endif()
