
libkernel_generate_bitcode_target(
    TARGETNAME amdgpu-amdhsa 
    TRIPLE amdgcn-amd-amdhsa
    SOURCES core.cpp math.cpp localmem.cpp
    ADDITIONAL_ARGS -nogpulib)
