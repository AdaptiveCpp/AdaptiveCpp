# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MultigridSYCL)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set a default build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)

  # Optionally list supported values for IDE GUIs
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


#if(NOT DEFINED CMAKE_CXX_COMPILER)
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUGMODE -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_COMPILER "acpp")

set(DEBUG_LIBS boost_stacktrace_addr2line dl backtrace)
set(LINK_OPS "-fno-pie")
set(COMP_OPS "-no-pie")


add_library(tracer_lib SHARED tracer_lib.cc)
target_compile_definitions(tracer_lib PRIVATE BOOST_STACKTRCAE_USE_ADDR2LINE)
target_link_libraries(tracer_lib PRIVATE ${DEBUG_LIBS})

add_library(mem_leak SHARED mem_leak_lib.cc)
target_compile_definitions(mem_leak PRIVATE BOOST_STACKTRCAE_USE_ADDR2LINE)
target_link_libraries(mem_leak PRIVATE ${DEBUG_LIBS})


add_library(event_leak SHARED event_leak_lib.cc)
target_compile_definitions(event_leak PRIVATE BOOST_STACKTRCAE_USE_ADDR2LINE)
target_link_libraries(event_leak PRIVATE ${DEBUG_LIBS})

add_library(print_dag SHARED print_dag_nodes.cc)
target_compile_definitions(event_leak PRIVATE BOOST_STACKTRCAE_USE_ADDR2LINE)
target_link_libraries(event_leak PRIVATE ${DEBUG_LIBS})





add_executable(first_trial first_trial.cc)
target_compile_definitions(first_trial PRIVATE BOOST_STACKTRCAE_USE_ADDR2LINE)
target_link_libraries(first_trial PRIVATE ${DEBUG_LIBS})
target_link_options(first_trial PRIVATE ${COMP_OPS})
target_compile_options(first_trial PRIVATE ${LINK_OPS})







message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Optional: Link additional libraries if needed
# target_link_libraries(YourExecutableName YourLibraryName)
