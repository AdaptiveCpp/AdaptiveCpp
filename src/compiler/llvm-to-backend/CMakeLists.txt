
set(LLVM_TO_BACKEND_INCLUDE_DIRS
  ../../../include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ACPP_BINARY_ROOT}/include)

set(CMAKE_INSTALL_RPATH ${base} ${base}/../../)

function(create_llvm_based_library)
  set(options STATIC)
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

  if(CREATE_LLVM_LIBRARY_STATIC)
    add_library(${target} STATIC
      ${sources})
  else()
    add_library(${target} SHARED
      ${sources})
  endif()

  target_include_directories(${target} PRIVATE
    ${LLVM_TO_BACKEND_INCLUDE_DIRS})
  target_include_directories(${target} SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(${target} PRIVATE
    ${LLVM_DEFINITIONS_LIST})
  target_compile_definitions(${target} PRIVATE -DHIPSYCL_COMPILER_COMPONENT)

  if(NOT ACPP_LLVM_COMPONENT)
    find_library(LLVM_LIBRARY NAMES LLVM LLVM-${LLVM_VERSION_MAJOR} HINTS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    if(NOT LLVM_LIBRARY)
      message(FATAL_ERROR "LLVM at ${LLVM_DIR} does not have libLLVM.so. Disable advanced compiler-based functionality (-DACPP_COMPILER_FEATURE_PROFILE=minimal, not recommended for most use cases) or choose another LLVM installation")
    endif()
  elseif(NOT LLVM_BUILD_LLVM_DYLIB AND NOT WIN32)
    message(FATAL_ERROR "LLVM is not configured to build libLLVM.so. Please disable reconfigure with LLVM_BUILD_LLVM_DYLIB and LLVM_LINK_LLVM_DYLIB.")
  endif()
  if(WIN32)
    llvm_config(${target} core support irreader passes)
  else()
    llvm_config(${target} USE_SHARED core support irreader passes)
  endif()
  # We need symbolic functions for stdpar
  target_link_libraries(${target} PRIVATE ${HIPSYCL_STDPAR_RT_LINKER_FLAGS})

  if(NOT LLVM_ENABLE_EH)
    target_compile_options(${target} PRIVATE -fno-exceptions)
  endif()
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(${target} PRIVATE -fno-rtti)
  endif()

  if(ACPP_LLVM_COMPONENT)
    add_dependencies(${target} intrinsics_gen)
  endif()

  set(install_subdir hipSYCL/llvm-to-backend)
  install(TARGETS ${target}
    RUNTIME DESTINATION bin/${install_subdir}
    LIBRARY DESTINATION lib/${install_subdir}
    ARCHIVE DESTINATION lib/${install_subdir})
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

  if(WIN32)
  set(MODE STATIC)
  endif()
  create_llvm_based_library(TARGET ${target} ${MODE} SOURCES ${sources})
  target_link_libraries(${target} PUBLIC llvm-to-backend)
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
  target_include_directories(${target}-tool SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  target_compile_definitions(${target}-tool PRIVATE
    ${LLVM_DEFINITIONS_LIST})
  target_compile_definitions(${target}-tool PRIVATE -DHIPSYCL_TOOL_COMPONENT)
  target_link_libraries(${target}-tool PRIVATE ${target})

  set(install_subdir hipSYCL/llvm-to-backend)
  install(TARGETS ${target}-tool
    RUNTIME DESTINATION bin/${install_subdir}
    LIBRARY DESTINATION lib/${install_subdir}
    ARCHIVE DESTINATION lib/${install_subdir})
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
  if(WIN32)
    set(MODE STATIC)
  endif()
  create_llvm_based_library(
    ${MODE}
    TARGET llvm-to-backend
    SOURCES
      LLVMToBackend.cpp
      AddressSpaceInferencePass.cpp
      KnownGroupSizeOptPass.cpp
      KnownPtrParamAlignmentOptPass.cpp
      GlobalSizesFitInI32OptPass.cpp
      GlobalInliningAttributorPass.cpp
      DeadArgumentEliminationPass.cpp
      ProcessS2ReflectionPass.cpp
      ../sscp/KernelOutliningPass.cpp
      Utils.cpp
      )

  target_compile_definitions(llvm-to-backend PRIVATE
    -DACPP_CLANG_PATH="${CLANG_INSTALLED_PATH}"
  )
  target_link_libraries(llvm-to-backend PUBLIC
    acpp-common
  )

  if(WITH_LLVM_TO_SPIRV)
    add_hipsycl_llvm_backend(
      BACKEND spirv
      LIBRARY spirv/LLVMToSpirv.cpp
      TOOL spirv/LLVMToSpirvTool.cpp)

    # Install LLVM-SPIRV translator for llvm-to-spirv
    if(NOT LLVMSPIRV_BRANCH)
      set(LLVMSPIRV_BRANCH llvm_release_${LLVM_VERSION_MAJOR}0)
    endif()
    set(LLVMSPIRV_RELATIVE_INSTALLDIR lib/hipSYCL/ext/llvm-spirv)
    set(LLVMSPIRV_INSTALLDIR ${CMAKE_INSTALL_PREFIX}/${LLVMSPIRV_RELATIVE_INSTALLDIR})
    set(LLVMSPIRV_PATH ${LLVMSPIRV_INSTALLDIR}/bin/llvm-spirv)
    set(LLVMSPIRV_RELATIVE_PATH ${LLVMSPIRV_RELATIVE_INSTALLDIR}/bin/llvm-spirv)

    ExternalProject_Add(LLVMSpirvTranslator
      GIT_REPOSITORY https://github.com/AdaptiveCpp/SPIRV-LLVM-Translator
      GIT_TAG origin/${LLVMSPIRV_BRANCH}
      GIT_SHALLOW ON
      GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
      BUILD_ALWAYS OFF
      BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/ext/llvm-spirv
      CMAKE_ARGS
        -DCMAKE_C_COMPILER:PATH=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER:PATH=${CMAKE_CXX_COMPILER}
      CMAKE_CACHE_ARGS
        -DLLVM_SPIRV_BUILD_EXTERNAL:BOOL=YES
        -DLLVM_DIR:PATH=${LLVM_DIR}
        -DCMAKE_INSTALL_PREFIX:PATH=${LLVMSPIRV_INSTALLDIR}
        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
      BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel ${ACPP_SUBPROJECT_PARALLEL_JOBS}
      INSTALL_COMMAND ""
    )

    add_custom_target(InstallLLVMSpirvTranslator
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/ext/llvm-spirv --target install
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ext/llvm-spirv
    )

    #install(TARGETS InstallLLVMSpirvTranslator)
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build . --target InstallLLVMSpirvTranslator)")

    add_dependencies(llvm-to-spirv LLVMSpirvTranslator)
    target_compile_definitions(llvm-to-spirv PRIVATE
      -DHIPSYCL_RELATIVE_LLVMSPIRV_PATH="${LLVMSPIRV_RELATIVE_PATH}")


  endif()

  if(WITH_LLVM_TO_PTX)
    add_hipsycl_llvm_backend(
      BACKEND ptx
      LIBRARY ptx/LLVMToPtx.cpp
      TOOL ptx/LLVMToPtxTool.cpp)

    target_compile_definitions(llvm-to-ptx PRIVATE
      -DHIPSYCL_CUDA_PATH="${CUDA_TOOLKIT_ROOT_DIR}"
      -DACPP_CUDA_DEVICE_LIBS_PATH="${CUDA_DEVICE_LIBS_PATH}")
  endif()

  if(WITH_LLVM_TO_AMDGPU_AMDHSA)
    add_hipsycl_llvm_backend(
      BACKEND amdgpu
      LIBRARY amdgpu/LLVMToAmdgpu.cpp
      TOOL amdgpu/LLVMToAmdgpuTool.cpp)

    target_compile_definitions(llvm-to-amdgpu PRIVATE
      -DACPP_ROCM_PATH="${ROCM_PATH}"
      -DACPP_ROCM_DEVICE_LIBS_PATH="${ROCM_DEVICE_LIBS_PATH}")

    find_program(HIPCC_PATH hipcc HINTS ${ROCM_PATH}/bin)
    if(HIPCC_PATH)
      target_compile_definitions(llvm-to-amdgpu PRIVATE
        -DACPP_HIPCC_PATH="${HIPCC_PATH}")
    endif()

    find_library(HIPRTC_LIBRARY hiprtc HINTS ${ROCM_PATH}/lib)

    if(HIPRTC_LIBRARY)
      message(STATUS "Found hipRTC: ${HIPRTC_LIBRARY}")

      target_include_directories(llvm-to-amdgpu PRIVATE ${ROCM_PATH}/include)
      target_link_libraries(llvm-to-amdgpu PRIVATE ${HIPRTC_LIBRARY})
      target_compile_definitions(llvm-to-amdgpu PRIVATE -DACPP_HIPRTC_LINK)
    endif()

  endif()


  if(WITH_LLVM_TO_HOST)
    include(CheckCXXCompilerFlag)

    # TODO We assume that if clang -mcpu/march=native works, llc -mcpu=native works too.
    # We should actually check llc/opt here!
    if(NOT ACPP_HOST_FORCE_MCPU_TARGET)
      check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
      check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)

      if(HAS_MCPU_NATIVE OR HAS_MARCH_NATIVE)
        set(HOST_LLC_CPU_FLAG -mcpu=native)
        set(HOST_OPT_CPU_FLAG --mcpu=native)
      else()
        set(HOST_LLC_CPU_FLAG "")
        set(HOST_OPT_CPU_FLAG "")
        message(SEND_ERROR "Could not determine that llc -mcpu=native works. Please provide the CPU that AdaptiveCpp should target at JIT-time using -DACPP_HOST_FORCE_MCPU_TARGET")
      endif()
    else()
      set(HOST_LLC_CPU_FLAG "-mcpu=${ACPP_HOST_FORCE_MCPU_TARGET}")
      set(HOST_OPT_CPU_FLAG "--mcpu=${ACPP_HOST_FORCE_MCPU_TARGET}")
    endif()
    message(STATUS "llvm-to-host: Using llc compilation flag to target native CPU: ${HOST_LLC_CPU_FLAG}")

    add_hipsycl_llvm_backend(
      BACKEND host
      LIBRARY host/LLVMToHost.cpp host/HostKernelWrapperPass.cpp host/StaticLocalMemoryPass.cpp
      TOOL host/LLVMToHostTool.cpp)

    target_compile_definitions(llvm-to-host PRIVATE
      -DACPP_LLC_HOST_CPU_FLAG="${HOST_LLC_CPU_FLAG}"
      -DACPP_OPT_HOST_CPU_FLAG="${HOST_OPT_CPU_FLAG}"
      -DACPP_LLC_ADDITIONAL_FLAGS="${HOST_ADDITIONAL_LLC_FLAGS}"
      -DACPP_OPT_ADDITIONAL_FLAGS="${HOST_ADDITIONAL_OPT_FLAGS}")
    target_link_libraries(llvm-to-host PRIVATE acpp-clang-cbs)
  endif()

  if(WITH_LLVM_TO_METAL)
    add_hipsycl_llvm_backend(
      BACKEND metal
      LIBRARY
        metal/LLVMToMetal.cpp
        metal/HLExtractionPass.cpp
        metal/HLTree.cpp
        metal/Emitter.cpp
      TOOL metal/LLVMToMetalTool.cpp)
  endif()

endif()
