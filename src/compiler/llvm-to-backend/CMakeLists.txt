
if(WITH_SSCP_COMPILER)

  add_library(llvm-to-spirv SHARED
    spirv/LLVMToSpirv.cpp)
  target_include_directories(llvm-to-spirv PRIVATE
    ../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include)
  
  target_compile_definitions(llvm-to-spirv PRIVATE ${LLVM_DEFINITIONS} -DHIPSYCL_COMPILER_COMPONENT)
  llvm_config(llvm-to-spirv core support irreader)

  add_executable(llvm-to-spirv-tool spirv/LLVMToSpirvTool.cpp)
  target_compile_definitions(llvm-to-spirv-tool PRIVATE -DHIPSYCL_COMPILER_COMPONENT)
  target_include_directories(llvm-to-spirv-tool PRIVATE
    ../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include)
  target_link_libraries(llvm-to-spirv-tool llvm-to-spirv)

  install(TARGETS llvm-to-spirv DESTINATION lib/hipSYCL/llvm-to-backend)
  install(TARGETS llvm-to-spirv-tool DESTINATION lib/hipSYCL/llvm-to-backend)
endif()
