/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "hipSYCL/common/config.hpp"

#include "hipSYCL/compiler/FrontendPlugin.hpp"
#include "hipSYCL/compiler/GlobalsPruningPass.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"


#ifdef HIPSYCL_WITH_STDPAR_COMPILER
#include "hipSYCL/compiler/stdpar/MallocToUSM.hpp"
#include "hipSYCL/compiler/stdpar/SyncElision.hpp"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#endif

#ifdef HIPSYCL_WITH_ACCELERATED_CPU
#include "hipSYCL/compiler/cbs/LoopsParallelMarker.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#endif

#ifdef HIPSYCL_WITH_SSCP_COMPILER
#include "hipSYCL/compiler/sscp/TargetSeparationPass.hpp"
#endif

#include "clang/Frontend/FrontendPluginRegistry.h"

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

#if LLVM_VERSION_MAJOR < 16
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#endif

namespace hipsycl {
namespace compiler {

static llvm::cl::opt<bool> EnableLLVMSSCP{
    "hipsycl-sscp", llvm::cl::init(false),
    llvm::cl::desc{"Enable hipSYCL LLVM SSCP compilation flow"}};

static llvm::cl::opt<bool> EnableStdPar{
    "hipsycl-stdpar", llvm::cl::init(false),
    llvm::cl::desc{"Enable hipSYCL C++ standard parallelism support"}};

static llvm::cl::opt<bool> StdparNoMallocToUSM{
    "hipsycl-stdpar-no-malloc-to-usm", llvm::cl::init(false),
    llvm::cl::desc{"Disable hipSYCL C++ standard parallelism malloc-to-usm compiler-side support"}};

// Register and activate passes

static clang::FrontendPluginRegistry::Add<hipsycl::compiler::FrontendASTAction>
    HipsyclFrontendPlugin{"hipsycl_frontend", "enable hipSYCL frontend action"};

#if LLVM_VERSION_MAJOR < 16
static void registerGlobalsPruningPass(const llvm::PassManagerBuilder &,
                                       llvm::legacy::PassManagerBase &PM) {
  PM.add(new GlobalsPruningPassLegacy{});
}

static llvm::RegisterStandardPasses
    RegisterGlobalsPruningPassOptLevel0(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                                        registerGlobalsPruningPass);

static llvm::RegisterStandardPasses
    RegisterGlobalsPruningPassOptimizerLast(llvm::PassManagerBuilder::EP_OptimizerLast,
                                            registerGlobalsPruningPass);

#ifdef HIPSYCL_WITH_ACCELERATED_CPU
static llvm::RegisterPass<SplitterAnnotationAnalysisLegacy>
    splitterAnnotationReg("splitter-annot-ana", "hipSYCL splitter annotation analysis pass",
                          true /* Only looks at CFG */, true /* Analysis Pass */);

static void registerLoopSplitAtBarrierPasses(const llvm::PassManagerBuilder &,
                                             llvm::legacy::PassManagerBase &PM) {
  registerCBSPipelineLegacy(PM);
}

static llvm::RegisterStandardPasses
    RegisterLoopSplitAtBarrierPassOptimizerFirst(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                                                 registerLoopSplitAtBarrierPasses);

static void registerMarkParallelPass(const llvm::PassManagerBuilder &,
                                     llvm::legacy::PassManagerBase &PM) {
  PM.add(new LoopsParallelMarkerPassLegacy());
}

// SROA adds loads / stores without adopting the llvm.access.group MD, need to re-add.
static llvm::RegisterStandardPasses
    RegisterMarkParallelBeforeVectorizer(llvm::PassManagerBuilder::EP_VectorizerStart,
                                         registerMarkParallelPass);
#endif // HIPSYCL_WITH_ACCELERATED_CPU
#endif // LLVM_VERSION_MAJOR < 16

#if !defined(_WIN32) && LLVM_VERSION_MAJOR >= 11
#define HIPSYCL_RESOLVE_AND_QUOTE(V) #V
#define HIPSYCL_STRINGIFY(V) HIPSYCL_RESOLVE_AND_QUOTE(V)
#define HIPSYCL_PLUGIN_VERSION_STRING                                                              \
  "v" HIPSYCL_STRINGIFY(HIPSYCL_VERSION_MAJOR) "." HIPSYCL_STRINGIFY(                              \
      HIPSYCL_VERSION_MINOR) "." HIPSYCL_STRINGIFY(HIPSYCL_VERSION_PATCH)

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "hipSYCL Clang plugin", HIPSYCL_PLUGIN_VERSION_STRING,
        [](llvm::PassBuilder &PB) {
          // Note: for Clang < 12, this EP is not called for O0, but the new PM isn't
          // really used there anyways..
          PB.registerOptimizerLastEPCallback([](llvm::ModulePassManager &MPM, OptLevel) {
            MPM.addPass(hipsycl::compiler::GlobalsPruningPass{});
          });

#ifdef HIPSYCL_WITH_STDPAR_COMPILER
          if(EnableStdPar) {
            if(!StdparNoMallocToUSM) {
              PB.registerPipelineStartEPCallback([&](llvm::ModulePassManager &MPM, OptLevel Level) {
                MPM.addPass(MallocToUSMPass{});
              });
            }
            PB.registerOptimizerLastEPCallback([&](llvm::ModulePassManager &MPM, OptLevel Level) {
              MPM.addPass(SyncElisionInliningPass{});
              MPM.addPass(llvm::AlwaysInlinerPass{});
              MPM.addPass(SyncElisionPass{});
            });
          }
#endif

#ifdef HIPSYCL_WITH_SSCP_COMPILER
          if(EnableLLVMSSCP){
            PB.registerPipelineStartEPCallback(
                [&](llvm::ModulePassManager &MPM, OptLevel Level) {
                  MPM.addPass(TargetSeparationPass{});
                });
          }
#endif


#ifdef HIPSYCL_WITH_ACCELERATED_CPU
          PB.registerAnalysisRegistrationCallback([](llvm::ModuleAnalysisManager &MAM) {
            MAM.registerPass([] { return SplitterAnnotationAnalysis{}; });
          });
#if LLVM_VERSION_MAJOR < 12
          PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM) {
            OptLevel Opt = OptLevel::O3;
#else
          PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM, OptLevel Opt) {
#endif
            registerCBSPipeline(MPM, Opt);
          });
          // SROA adds loads / stores without adopting the llvm.access.group MD, need to re-add.
          // todo: check back with LLVM 13, might be fixed with https://reviews.llvm.org/D103254
          PB.registerVectorizerStartEPCallback([](llvm::FunctionPassManager &FPM, OptLevel) {
            FPM.addPass(LoopsParallelMarkerPass{});
          });
#endif
        }
  };
}
#endif // !_WIN32 && LLVM_VERSION_MAJOR >= 11

} // namespace compiler
} // namespace hipsycl
