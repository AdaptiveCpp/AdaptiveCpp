# First build our custom compiler launcher
execute_process(COMMAND ${CMAKE_CXX_COMPILER} 
                        ${CMAKE_CURRENT_SOURCE_DIR}/compiler_launcher.cc 
                        -o ${CMAKE_BINARY_DIR}/compiler_launcher/compiler_launcher)

set(CMAKE_CXX_COMPILER_LAUNCHER ${CMAKE_BINARY_DIR}/compiler_launcher/compiler_launcher)

add_executable(compiler_launcher_test main.cc)
add_sycl_to_target(TARGET compiler_launcher_test)
