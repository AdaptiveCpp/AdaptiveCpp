cmake_minimum_required(VERSION 3.5)
project(opensycl-clang)

get_filename_component(CLANG_BINARY_PREFIX ${CLANG_EXECUTABLE_PATH} DIRECTORY)

if(NOT HIPSYCL_DEBUG_LEVEL)
  set(HIPSYCL_DEBUG_LEVEL 2 CACHE INTEGER)
endif()

add_definitions(-DHIPSYCL_DEBUG_LEVEL=${HIPSYCL_DEBUG_LEVEL})
if(WITH_ACCELERATED_CPU)
  add_definitions(-DHIPSYCL_WITH_ACCELERATED_CPU)
endif()

if(WITH_ACCELERATED_CPU)
  set(CBS_PLUGIN cbs/LoopSplitterInlining.cpp
    cbs/SplitterAnnotationAnalysis.cpp
    cbs/IRUtils.cpp
    cbs/KernelFlattening.cpp
    cbs/LoopsParallelMarker.cpp
    cbs/PHIsToAllocas.cpp
    cbs/RemoveBarrierCalls.cpp
    cbs/CanonicalizeBarriers.cpp
    cbs/SimplifyKernel.cpp
    cbs/LoopSimplify.cpp
    cbs/PipelineBuilder.cpp
    cbs/SubCfgFormation.cpp
    cbs/UniformityAnalysis.cpp
    cbs/VectorShape.cpp
    cbs/VectorizationInfo.cpp
    cbs/AllocaSSA.cpp
    cbs/VectorShapeTransformer.cpp
    cbs/Region.cpp
    cbs/SyncDependenceAnalysis.cpp)
else()
  set(CBS_PLUGIN "")
endif()

if(WITH_SSCP_COMPILER)
  set(SSCP_COMPILER 
    sscp/KernelOutliningPass.cpp
    sscp/IRConstantReplacer.cpp
    sscp/TargetSeparationPass.cpp
    sscp/HostKernelNameExtractionPass.cpp
    sscp/AggregateArgumentExpansionPass.cpp
    sscp/StdBuiltinRemapperPass.cpp)
  add_definitions(-DHIPSYCL_WITH_SSCP_COMPILER)
else()
  set(SSCP_COMPILER "")
endif()

if(WITH_STDPAR_COMPILER)
  set(STDPAR_COMPILER
    stdpar/MallocToUSM.cpp
    stdpar/SyncElision.cpp)
    add_definitions(-DHIPSYCL_WITH_STDPAR_COMPILER)
else()
  set(STDPAR_COMPILER "")
endif()

add_library(opensycl-clang SHARED
  OpenSYCLClangPlugin.cpp
  GlobalsPruningPass.cpp
  ${CBS_PLUGIN}
  ${SSCP_COMPILER}
  ${STDPAR_COMPILER}
)

target_include_directories(opensycl-clang PRIVATE
  ../../include
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_BINARY_DIR}/include)

target_compile_definitions(opensycl-clang PRIVATE
  ${LLVM_DEFINITIONS} -DHIPSYCL_COMPILER_COMPONENT)

if(ROCM_VERSION_MAJOR)
  target_compile_definitions(opensycl-clang PRIVATE -DROCM_CLANG_VERSION_MAJOR=${ROCM_VERSION_MAJOR} -DROCM_CLANG_VERSION_MINOR=${ROCM_VERSION_MINOR} -DROCM_CLANG_VERSION_PATCH=${ROCM_VERSION_PATCH})
endif()
if(HIPSYCL_NO_DEVICE_MANGLER)
  target_compile_definitions(opensycl-clang PRIVATE -DHIPSYCL_NO_DEVICE_MANGLER)
endif()

if(NOT ${LLVM_ENABLE_EH})
  target_compile_options(opensycl-clang PRIVATE -fno-exceptions)
endif()

if(NOT ${LLVM_ENABLE_RTTI})
  target_compile_options(opensycl-clang PRIVATE -fno-rtti)
endif()

target_link_libraries(opensycl-clang
  ${LLVM_LIBS})
if(APPLE)
  target_link_libraries(opensycl-clang -Wl,-undefined -Wl,dynamic_lookup)
elseif(WIN32)
  set(CLANG_LIBRARY_DIR ${LLVM_PREFIX_DIR}/lib/)
  target_link_directories(opensycl-clang PRIVATE ${CLANG_LIBRARY_DIR})
  target_link_libraries(opensycl-clang clang LLVMSupport)
  set_target_properties(opensycl-clang PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS On)
else()
  target_link_libraries(opensycl-clang -Wl,-znodelete)
endif()

install(TARGETS opensycl-clang
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib)

add_subdirectory(llvm-to-backend)
