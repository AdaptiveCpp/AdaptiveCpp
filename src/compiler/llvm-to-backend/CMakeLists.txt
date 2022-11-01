
set(LLVM_TO_BACKEND_INCLUDE_DIRS 
  ../../../include
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include)


function(create_llvm_based_library)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(CREATE_LLVM_LIBRARY
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CREATE_LLVM_LIBRARY_TARGET})
  set(sources ${CREATE_LLVM_LIBRARY_SOURCES})

  add_library(${target} SHARED
    ${sources})

  target_include_directories(${target} PRIVATE
    ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  
  target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS} -DHIPSYCL_COMPILER_COMPONENT)
  llvm_config(${target} USE_SHARED core support irreader passes)

  install(TARGETS ${target} DESTINATION lib/hipSYCL/llvm-to-backend)
endfunction()

function(create_llvm_to_backend_library)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(CREATE_LLVM_LIBRARY
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CREATE_LLVM_LIBRARY_TARGET})
  set(sources ${CREATE_LLVM_LIBRARY_SOURCES})

  create_llvm_based_library(TARGET ${target} SOURCES ${sources})
  target_link_libraries(${target} PUBLIC llvm-to-backend hipSYCL-common)
endfunction()

function(create_llvm_to_backend_tool)
  set(options)
  set(one_value_keywords TARGET)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(CREATE_LLVM_TOOL
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(target ${CREATE_LLVM_TOOL_TARGET})
  set(sources ${CREATE_LLVM_TOOL_SOURCES})
 
  add_executable(${target}-tool ${sources})

  target_include_directories(${target}-tool PRIVATE
    ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  
  target_compile_definitions(${target}-tool PRIVATE ${LLVM_DEFINITIONS} -DHIPSYCL_TOOL_COMPONENT)
  target_link_libraries(${target}-tool PRIVATE ${target})

  install(TARGETS ${target}-tool DESTINATION lib/hipSYCL/llvm-to-backend)
endfunction()

function(add_hipsycl_llvm_backend)
  set(options)
  set(one_value_keywords BACKEND)
  set(multi_value_keywords LIBRARY TOOL)
  cmake_parse_arguments(ADD_BACKEND
    "${options}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  set(backend_name ${ADD_BACKEND_BACKEND})
  set(tool_sources ${ADD_BACKEND_TOOL})
  set(library_sources ${ADD_BACKEND_LIBRARY})

  create_llvm_to_backend_library(TARGET llvm-to-${backend_name} SOURCES ${library_sources})
  create_llvm_to_backend_tool(TARGET llvm-to-${backend_name} SOURCES ${tool_sources})
endfunction()

include(ExternalProject)

if(WITH_SSCP_COMPILER)
  create_llvm_based_library(
    TARGET llvm-to-backend
    SOURCES LLVMToBackend.cpp ../sscp/KernelOutliningPass.cpp)

  add_hipsycl_llvm_backend(
    BACKEND spirv 
    LIBRARY spirv/LLVMToSpirv.cpp 
    TOOL spirv/LLVMToSpirvTool.cpp)

  add_hipsycl_llvm_backend(
    BACKEND ptx 
    LIBRARY ptx/LLVMToPtx.cpp 
    TOOL ptx/LLVMToPtxTool.cpp)

# Install LLVM-SPIRV translator for llvm-to-spirv
  set(LLVMSPIRV_BRANCH llvm_release_${LLVM_VERSION_MAJOR}0)
  set(LLVMSPIRV_RELATIVE_INSTALLDIR lib/hipSYCL/ext/llvm-spirv)
  set(LLVMSPIRV_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/${LLVMSPIRV_RELATIVE_INSTALLDIR})
  set(LLVMSPIRV_PATH ${LLVMSPIRV_INSTALLDIR}/bin/llvm-spirv)
  set(LLVMSPIRV_RELATIVE_PATH ${LLVMSPIRV_RELATIVE_INSTALLDIR}/bin/llvm-spirv)

  ExternalProject_Add(LLVMSpirvTranslator
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-LLVM-Translator
    GIT_TAG origin/${LLVMSPIRV_BRANCH}
    GIT_SHALLOW ON
    BUILD_ALWAYS OFF
    INSTALL_DIR ${LLVMSPIRV_INSTALLDIR}
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext/llvm-spirv
    CMAKE_CACHE_ARGS
      -DLLVM_SPIRV_BUILD_EXTERNAL:BOOL=YES
      -DLLVM_DIR:PATH=${LLVM_DIR}
		  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target install
    )
  
  target_compile_definitions(llvm-to-spirv PRIVATE -DHIPSYCL_RELATIVE_LLVMSPIRV_PATH="${LLVMSPIRV_RELATIVE_PATH}")
endif()
