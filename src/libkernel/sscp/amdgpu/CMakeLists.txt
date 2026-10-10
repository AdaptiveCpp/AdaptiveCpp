if(WITH_LLVM_TO_AMDGPU_AMDHSA)
  set(amdgpu_libkernel_additional_args -nogpulib)

  if(WIN32)
    list(APPEND amdgpu_libkernel_additional_args
        -Xclang -fwchar-type=short)
  endif()

  set(GENERATED_HPP "${HIPSYCL_SOURCE_DIR}/include/hipSYCL/sycl/libkernel/sscp/builtins/amdgpu_auto_builtins.hpp")
  set(GENERATED_CPP "${CMAKE_CURRENT_SOURCE_DIR}/amdgpu_auto_builtins.cpp")

  add_custom_command(
      OUTPUT ${GENERATED_HPP} ${GENERATED_CPP}
      COMMAND acpp-amdgpu-builtin-gen ${GENERATED_HPP} ${GENERATED_CPP}
      DEPENDS acpp-amdgpu-builtin-gen
      COMMENT "Auto-generating AMDGPU builtin wrappers via LibTooling..."
  )

  # Create a custom target that forces this generation to happen early
  add_custom_target(generate_amdgpu_builtins ALL DEPENDS ${GENERATED_HPP} ${GENERATED_CPP})

  set_source_files_properties(${GENERATED_CPP} PROPERTIES GENERATED TRUE)

  libkernel_generate_bitcode_target(
      TARGETNAME amdgpu-amdhsa 
      TRIPLE amdgcn-amd-amdhsa
      SOURCES 
      amdgpu_auto_builtins.cpp
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
