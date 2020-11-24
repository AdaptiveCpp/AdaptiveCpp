cmake_minimum_required(VERSION 3.5)
project(hipSYCL_clang)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


get_filename_component(CLANG_BINARY_PREFIX ${CLANG_EXECUTABLE_PATH} DIRECTORY)

if(NOT HIPSYCL_DEBUG_LEVEL)
  set(HIPSYCL_DEBUG_LEVEL 1 CACHE INTEGER)
endif()

add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})

add_library(hipSYCL_clang SHARED
  HipsyclClangPlugin.cpp)

target_include_directories(hipSYCL_clang PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ../../include)

target_compile_definitions(hipSYCL_clang PRIVATE
  ${LLVM_DEFINITIONS} -DHIPSYCL_COMPILER_COMPONENT)
  
if(NOT ${LLVM_ENABLE_EH})
  target_compile_options(hipSYCL_clang PRIVATE -fno-exceptions)
endif()

if(NOT ${LLVM_ENABLE_RTTI})
  target_compile_options(hipSYCL_clang PRIVATE -fno-rtti)
endif()

target_link_libraries(hipSYCL_clang
  ${LLVM_LIBS}
  ${Boost_LIBRARIES})
if(APPLE)
    target_link_libraries(hipSYCL_clang -Wl,-undefined -Wl,dynamic_lookup)
else()
    target_link_libraries(hipSYCL_clang -Wl,-znodelete)
endif()

install(TARGETS hipSYCL_clang
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
