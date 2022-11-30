
libkernel_generate_bitcode_target(
    TARGETNAME amdgpu-amdhsa 
    TRIPLE amdgcn-amd-amdhsa
    SOURCES core.cpp integer.cpp math.cpp native.cpp relational.cpp subgroup.cpp localmem.cpp
    ADDITIONAL_ARGS -nogpulib)
