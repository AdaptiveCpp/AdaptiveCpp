
libkernel_generate_bitcode_target(
    TARGETNAME ptx 
    TRIPLE nvptx64-nvidia-cuda
    SOURCES core.cpp integer.cpp relational.cpp math.cpp localmem.cpp)
