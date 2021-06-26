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
#include "hipSYCL/compiler/BarrierTailReplication.hpp"
#include "hipSYCL/compiler/CanonicalizeBarriers.hpp"
#include "hipSYCL/compiler/FrontendPlugin.hpp"
#include "hipSYCL/compiler/IR.hpp"
#include "hipSYCL/compiler/IsolateRegions.hpp"
#include "hipSYCL/compiler/KernelFlattening.hpp"
#include "hipSYCL/compiler/LoopSplitter.hpp"
#include "hipSYCL/compiler/LoopSplitterInlining.hpp"
#include "hipSYCL/compiler/LoopsParallelMarker.hpp"
#include "hipSYCL/compiler/PHIsToAllocas.hpp"
#include "hipSYCL/compiler/RemoveBarrierCalls.hpp"
#include "hipSYCL/compiler/ReqdLoopBarriers.hpp"
#include "hipSYCL/compiler/SimplifyKernel.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"
#include "hipSYCL/compiler/WILoopMarker.hpp"
#include "hipSYCL/compiler/WorkItemLoopCreation.hpp"

#include "clang/Frontend/FrontendPluginRegistry.h"

#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

namespace hipsycl {
namespace compiler {
// Register and activate passes

static clang::FrontendPluginRegistry::Add<hipsycl::compiler::FrontendASTAction> HipsyclFrontendPlugin{
    "hipsycl_frontend", "enable hipSYCL frontend action"};

static void registerFunctionPruningIRPass(const llvm::PassManagerBuilder &, llvm::legacy::PassManagerBase &PM) {
  PM.add(new FunctionPruningIRPass{});
}

static llvm::RegisterStandardPasses
    RegisterFunctionPruningIRPassOptLevel0(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                                           registerFunctionPruningIRPass);

static llvm::RegisterStandardPasses
    RegisterFunctionPruningIRPassOptimizerLast(llvm::PassManagerBuilder::EP_OptimizerLast,
                                               registerFunctionPruningIRPass);

static llvm::RegisterPass<SplitterAnnotationAnalysisLegacy>
    splitterAnnotationReg("splitter-annot-ana", "hipSYCL splitter annotation analysis pass",
                          true /* Only looks at CFG */, true /* Analysis Pass */);

static llvm::RegisterPass<VariableUniformityAnalysisLegacy>
    varUniformityReg("var-uniformity", "hipSYCL variable uniformity analysis pass", true /* Only looks at CFG */,
                     true /* Analysis Pass */);

static void registerLoopSplitAtBarrierPassesO0(const llvm::PassManagerBuilder &, llvm::legacy::PassManagerBase &PM) {
  PM.add(new SimplifyKernelPassLegacy{});
  PM.add(new WILoopMarkerPassLegacy{});
  PM.add(new LoopSplitterInliningPassLegacy{});
  //  PM.add(new LoopSplitAtBarrierPassLegacy{true});
  PM.add(new PHIsToAllocasPassLegacy{});
  PM.add(new IsolateRegionsPassLegacy{});
  PM.add(new AddRequiredLoopBarriersPassLegacy{});
  PM.add(new BarrierTailReplicationPassLegacy{});
  PM.add(new CanonicalizeBarriersPassLegacy{});
  PM.add(new IsolateRegionsPassLegacy{});
  PM.add(new WorkItemLoopCreationPassLegacy{});
  PM.add(new RemoveBarrierCallsPassLegacy{});
}

static void registerLoopSplitAtBarrierPasses(const llvm::PassManagerBuilder &, llvm::legacy::PassManagerBase &PM) {
  PM.add(new SimplifyKernelPassLegacy{});
  PM.add(new WILoopMarkerPassLegacy{});
  PM.add(new LoopSplitterInliningPassLegacy{});
  //  PM.add(new LoopSplitAtBarrierPassLegacy{false});
  PM.add(new PHIsToAllocasPassLegacy{});
  PM.add(new IsolateRegionsPassLegacy{});
  PM.add(new AddRequiredLoopBarriersPassLegacy{});
  PM.add(new BarrierTailReplicationPassLegacy{});
  PM.add(new CanonicalizeBarriersPassLegacy{});
  PM.add(new IsolateRegionsPassLegacy{});
  PM.add(new WorkItemLoopCreationPassLegacy{});
  PM.add(new RemoveBarrierCallsPassLegacy{});
  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new LoopsParallelMarkerPassLegacy{});
}

static llvm::RegisterStandardPasses
    RegisterLoopSplitAtBarrierPassOptLevel0(llvm::PassManagerBuilder::EP_EnabledOnOptLevel0,
                                            registerLoopSplitAtBarrierPassesO0);

static llvm::RegisterStandardPasses
    RegisterLoopSplitAtBarrierPassOptimizerLast(llvm::PassManagerBuilder::EP_ModuleOptimizerEarly,
                                                registerLoopSplitAtBarrierPasses);

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "hipSYCL Clang plugin", "v0.9", [](llvm::PassBuilder &PB) {
      PB.registerAnalysisRegistrationCallback(
          [](llvm::ModuleAnalysisManager &MAM) { MAM.registerPass([] { return SplitterAnnotationAnalysis{}; }); });
      PB.registerAnalysisRegistrationCallback(
          [](llvm::FunctionAnalysisManager &FAM) { FAM.registerPass([] { return VariableUniformityAnalysis{}; }); });
#if LLVM_VERSION_MAJOR < 12
      PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM) {
#else
      PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM, llvm::PassBuilder::OptimizationLevel Opt) {
#endif
        MPM.addPass(SplitterAnnotationAnalysisCacher{});

        llvm::FunctionPassManager FPM;
        FPM.addPass(SimplifyKernelPass{});
        FPM.addPass(WILoopMarkerPass{});
        FPM.addPass(LoopSplitterInliningPass{});

        FPM.addPass(PHIsToAllocasPass{});
        FPM.addPass(IsolateRegionsPass{});

        llvm::LoopPassManager LPM;
        LPM.addPass(AddRequiredLoopBarriersPass{});
        FPM.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM)));

        FPM.addPass(BarrierTailReplicationPass{});
        FPM.addPass(CanonicalizeBarriersPass{});
        FPM.addPass(llvm::LoopSimplifyPass{});

        FPM.addPass(IsolateRegionsPass{});
        FPM.addPass(WorkItemLoopCreationPass{});
        FPM.addPass(RemoveBarrierCallsPass{});

        // todo: remove or integrate in legacy as well or add custom wrapper pass?
        FPM.addPass(llvm::LoopSimplifyPass{});

        // llvm::LoopPassManager LPM;
        // LPM.addPass(LoopSplitAtBarrierPass{true});
        // FPM.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM)));

#if LLVM_VERSION_MAJOR >= 12
        if (Opt == llvm::PassBuilder::OptimizationLevel::O3)
#endif
          FPM.addPass(KernelFlatteningPass{});
#if LLVM_VERSION_MAJOR >= 12
        if (Opt != llvm::PassBuilder::OptimizationLevel::O0)
#endif
          FPM.addPass(LoopsParallelMarkerPass{});
        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
      });
    }
  };
}

} // namespace compiler
} // namespace hipsycl
