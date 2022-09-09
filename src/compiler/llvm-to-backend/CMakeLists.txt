
if(WITH_SSCP_COMPILER)

  set(LLVM_TO_BACKEND_INCLUDE_DIRS 
    ../../../include
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include)
  
  add_library(llvm-to-spirv SHARED
    spirv/LLVMToSpirv.cpp)
  target_include_directories(llvm-to-spirv PRIVATE
    ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  
  target_compile_definitions(llvm-to-spirv PRIVATE ${LLVM_DEFINITIONS} -DHIPSYCL_COMPILER_COMPONENT)
  llvm_config(llvm-to-spirv core support irreader passes)

  add_executable(llvm-to-spirv-tool spirv/LLVMToSpirvTool.cpp)
  target_compile_definitions(llvm-to-spirv-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
  target_include_directories(llvm-to-spirv-tool PRIVATE
  ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  target_link_libraries(llvm-to-spirv-tool llvm-to-spirv)

  install(TARGETS llvm-to-spirv DESTINATION lib/hipSYCL/llvm-to-backend)
  install(TARGETS llvm-to-spirv-tool DESTINATION lib/hipSYCL/llvm-to-backend)
endif()
