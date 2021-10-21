/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
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

#include "hipSYCL/compiler/PipelineBuilder.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/BarrierTailReplication.hpp"
#include "hipSYCL/compiler/CanonicalizeBarriers.hpp"
#include "hipSYCL/compiler/IsolateRegions.hpp"
#include "hipSYCL/compiler/KernelFlattening.hpp"
#include "hipSYCL/compiler/LoopSimplify.hpp"
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
#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/Transforms/IPO/SCCP.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

#include <cstdlib>

namespace llvm {
FunctionPass *createSROAPass();
ModulePass *createIPSCCPPass();
#if LLVM_VERSION_MAJOR >= 12
FunctionPass *createCFGSimplificationPass(SimplifyCFGOptions Options,
                                          std::function<bool(const Function &)> Ftor = nullptr);
#else
FunctionPass *createCFGSimplificationPass(unsigned Threshold = 1, bool ForwardSwitchCond = false,
                                          bool ConvertSwitch = false, bool KeepLoops = true, bool SinkCommon = false,
                                          std::function<bool(const Function &)> Ftor = nullptr);
#endif
} // namespace llvm

namespace hipsycl::compiler {
LoopSplittingPipeline selectPipeline() {
  // for debugging purposes, it's useful to have the output flushed immediately
  llvm::outs().SetUnbuffered();

  const auto *Env = std::getenv("HIPSYCL_PIPELINE");
  if (!Env)
    return DefaultLoopSplittingPipeline;
  llvm::StringRef EnvVar{Env};
  if (EnvVar.contains("org"))
    return LoopSplittingPipeline::Original;
  if (EnvVar.contains("pocl"))
    return LoopSplittingPipeline::Pocl;
  if (EnvVar.contains("cbs"))
    return LoopSplittingPipeline::ContinuationBasedSynchronization;
  return DefaultLoopSplittingPipeline;
}

void registerOriginalPipelineLegacy(llvm::legacy::PassManagerBase &PM) {
  PM.add(new WILoopMarkerPassLegacy{});
  PM.add(new LoopSplitterInliningPassLegacy{});
  PM.add(new LoopSimplifyPassLegacy{});
  PM.add(new LoopSplitAtBarrierPassLegacy{true});
  PM.add(new RemoveBarrierCallsPassLegacy{});
  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new LoopsParallelMarkerPassLegacy{});
}
void registerPoclPipelineLegacy(llvm::legacy::PassManagerBase &PM) {
  PM.add(new WILoopMarkerPassLegacy{});
  PM.add(new LoopSplitterInliningPassLegacy{});
  PM.add(new SimplifyKernelPassLegacy{});

  PM.add(new PHIsToAllocasPassLegacy{});
  PM.add(new IsolateRegionsPassLegacy{});
  PM.add(new AddRequiredLoopBarriersPassLegacy{});
  PM.add(new BarrierTailReplicationPassLegacy{});
  PM.add(new PHIsToAllocasPassLegacy{});

  PM.add(new LoopSimplifyPassLegacy{});
  PM.add(new CanonicalizeBarriersPassLegacy{});

  PM.add(new IsolateRegionsPassLegacy{});
  PM.add(new WorkItemLoopCreationPassLegacy{});
  PM.add(new RemoveBarrierCallsPassLegacy{});
  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new LoopsParallelMarkerPassLegacy{});
}
void registerCBSPipelineLegacy(llvm::legacy::PassManagerBase &PM) {
  HIPSYCL_DEBUG_WARNING << "CBS pipeline might not result in peak performance with old PM\n";
  PM.add(new WILoopMarkerPassLegacy{});
  PM.add(new LoopSplitterInliningPassLegacy{});

  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new SimplifyKernelPassLegacy{});

  PM.add(new PHIsToAllocasPassLegacy{});
  PM.add(new LoopSimplifyPassLegacy{});

  PM.add(new CanonicalizeBarriersPassLegacy{});
  PM.add(new SubCfgFormationPassLegacy{});

  PM.add(new RemoveBarrierCallsPassLegacy{});

  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new LoopsParallelMarkerPassLegacy{});
}

void registerOriginalPipeline(llvm::ModulePassManager &MPM, OptLevel Opt) {
  MPM.addPass(SplitterAnnotationAnalysisCacher{});

  llvm::FunctionPassManager FPM;
  FPM.addPass(WILoopMarkerPass{});
  FPM.addPass(LoopSplitterInliningPass{});
  FPM.addPass(llvm::LoopSimplifyPass{});

  llvm::LoopPassManager LPM;
  LPM.addPass(LoopSplitAtBarrierPass{true});
  FPM.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM)));

  FPM.addPass(RemoveBarrierCallsPass{});

  if (Opt == OptLevel::O3)
    FPM.addPass(KernelFlatteningPass{});
  if (Opt != OptLevel::O0)
    FPM.addPass(LoopsParallelMarkerPass{});
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
}

void registerPoclPipeline(llvm::ModulePassManager &MPM, OptLevel Opt) {
  if (Opt == OptLevel::O0) {
    HIPSYCL_DEBUG_ERROR << "POCL pipeline is currently not fully supported with -O0 on the new PM\n";
  }
  MPM.addPass(SplitterAnnotationAnalysisCacher{});

  llvm::FunctionPassManager FPM;
  FPM.addPass(WILoopMarkerPass{});
  FPM.addPass(LoopSplitterInliningPass{});
  FPM.addPass(SimplifyKernelPass{});

  FPM.addPass(PHIsToAllocasPass{});
  FPM.addPass(IsolateRegionsPass{});

  llvm::LoopPassManager LPM;
  LPM.addPass(AddRequiredLoopBarriersPass{});
  FPM.addPass(llvm::createFunctionToLoopPassAdaptor(std::move(LPM)));

  FPM.addPass(BarrierTailReplicationPass{});
  FPM.addPass(PHIsToAllocasPass{});
  FPM.addPass(llvm::LoopSimplifyPass{});
  FPM.addPass(CanonicalizeBarriersPass{});

  FPM.addPass(IsolateRegionsPass{});
  FPM.addPass(WorkItemLoopCreationPass{});
  FPM.addPass(RemoveBarrierCallsPass{});

  FPM.addPass(llvm::LoopSimplifyPass{});

  if (Opt == OptLevel::O3)
    FPM.addPass(KernelFlatteningPass{});
  if (Opt != OptLevel::O0)
    FPM.addPass(LoopsParallelMarkerPass{});
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
}

void registerCBSPipeline(llvm::ModulePassManager &MPM, OptLevel Opt) {
  MPM.addPass(SplitterAnnotationAnalysisCacher{});

  llvm::FunctionPassManager FPM;
  FPM.addPass(WILoopMarkerPass{});
  FPM.addPass(LoopSplitterInliningPass{});

  FPM.addPass(KernelFlatteningPass{});
  FPM.addPass(SimplifyKernelPass{});

  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
  MPM.addPass(llvm::IPSCCPPass{});
  FPM.addPass(llvm::InstCombinePass{});
  FPM.addPass(llvm::SROA{});

  FPM.addPass(llvm::SimplifyCFGPass{});

  FPM.addPass(SimplifyKernelPass{});
#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
  FPM.addPass(PHIsToAllocasPass{});
#endif
  FPM.addPass(llvm::LoopSimplifyPass{});

  FPM.addPass(CanonicalizeBarriersPass{});

  FPM.addPass(SubCfgFormationPass{});

  FPM.addPass(RemoveBarrierCallsPass{});

  if (Opt == OptLevel::O3)
    FPM.addPass(KernelFlatteningPass{});
  if (Opt != OptLevel::O0)
    FPM.addPass(LoopsParallelMarkerPass{});
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
}

} // namespace hipsycl::compiler