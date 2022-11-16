
libkernel_generate_bitcode_target(
    TARGETNAME spirv 
    TRIPLE spir64-unknown-unknown
    SOURCES core.cpp math.cpp native.cpp integer.cpp relational.cpp localmem.cpp)
