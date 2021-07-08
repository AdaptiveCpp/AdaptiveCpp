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
#include "hipSYCL/compiler/FrontendPlugin.hpp"
#include "hipSYCL/compiler/IR.hpp"
#include "hipSYCL/compiler/PipelineBuilder.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

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

static void registerLoopSplitAtBarrierPasses(const llvm::PassManagerBuilder &, llvm::legacy::PassManagerBase &PM) {
  switch (hipsycl::compiler::selectPipeline()) {
  case LoopSplittingPipeline::Original:
    registerOriginalPipelineLegacy(PM);
    break;
  case hipsycl::compiler::LoopSplittingPipeline::Pocl:
    registerPoclPipelineLegacy(PM);
    break;
  case LoopSplittingPipeline::ContinuationBasedSynchronization:
    registerCBSPipelineLegacy(PM);
    break;
  }
}

static llvm::RegisterStandardPasses
    RegisterLoopSplitAtBarrierPassOptimizerLast(llvm::PassManagerBuilder::EP_EarlyAsPossible,
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
        llvm::PassBuilder::OptimizationLevel Opt = llvm::PassBuilder::OptimizationLevel::O3;
#else
      PB.registerPipelineStartEPCallback([](llvm::ModulePassManager &MPM, llvm::PassBuilder::OptimizationLevel Opt) {
#endif
        switch (hipsycl::compiler::selectPipeline()) {
        case LoopSplittingPipeline::Original:
          registerOriginalPipeline(MPM, Opt);
          break;
        case hipsycl::compiler::LoopSplittingPipeline::Pocl:
          registerPoclPipeline(MPM, Opt);
          break;
        case LoopSplittingPipeline::ContinuationBasedSynchronization:
          registerCBSPipeline(MPM, Opt);
          break;
        }
      });
    }
  };
}

} // namespace compiler
} // namespace hipsycl
