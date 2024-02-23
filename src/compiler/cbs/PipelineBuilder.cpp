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

#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/CanonicalizeBarriers.hpp"
#include "hipSYCL/compiler/cbs/KernelFlattening.hpp"
#include "hipSYCL/compiler/cbs/LoopSimplify.hpp"
#include "hipSYCL/compiler/cbs/LoopSplitterInlining.hpp"
#include "hipSYCL/compiler/cbs/LoopsParallelMarker.hpp"
#include "hipSYCL/compiler/cbs/PHIsToAllocas.hpp"
#include "hipSYCL/compiler/cbs/RemoveBarrierCalls.hpp"
#include "hipSYCL/compiler/cbs/SimplifyKernel.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/HostKernelWrapperPass.hpp"

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/SCCP.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>

namespace hipsycl::compiler {

#if LLVM_VERSION_MAJOR < 16
void registerCBSPipelineLegacy(llvm::legacy::PassManagerBase &PM) {
  HIPSYCL_DEBUG_WARNING << "CBS pipeline might not result in peak performance with old PM\n";
  PM.add(new LoopSplitterInliningPassLegacy{});

  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new SimplifyKernelPassLegacy{});
#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
  PM.add(new PHIsToAllocasPassLegacy{});
#endif

  PM.add(new LoopSimplifyPassLegacy{});

  PM.add(new CanonicalizeBarriersPassLegacy{});
  PM.add(new SubCfgFormationPassLegacy{});

  PM.add(new RemoveBarrierCallsPassLegacy{});

  PM.add(new KernelFlatteningPassLegacy{});
  PM.add(new LoopsParallelMarkerPassLegacy{});
}
#endif // LLVM_VERSION_MAJOR < 16

#if defined(ROCM_CLANG_VERSION_MAJOR) && ROCM_CLANG_VERSION_MAJOR == 5 && ROCM_CLANG_VERSION_MINOR == 5
#define IS_ROCM_CLANG_VERSION_5_5_0
#endif

void registerCBSPipeline(llvm::ModulePassManager &MPM, OptLevel Opt, bool IsSscp) {
  MPM.addPass(SplitterAnnotationAnalysisCacher{});

  llvm::FunctionPassManager FPM;
  FPM.addPass(LoopSplitterInliningPass{});

  if (Opt != OptLevel::O0) {
    FPM.addPass(KernelFlatteningPass{});
    FPM.addPass(SimplifyKernelPass{});

    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
    FPM = llvm::FunctionPassManager{};

    MPM.addPass(llvm::IPSCCPPass{});
    FPM.addPass(llvm::InstCombinePass{});
#if LLVM_VERSION_MAJOR <= 13
    FPM.addPass(llvm::SROA{});
#elif (LLVM_VERSION_MAJOR < 16) || defined(IS_ROCM_CLANG_VERSION_5_5_0)
    FPM.addPass(llvm::SROAPass{});
#else
    FPM.addPass(llvm::SROAPass{llvm::SROAOptions::ModifyCFG});
#endif

    FPM.addPass(llvm::SimplifyCFGPass{});
  }

  FPM.addPass(SimplifyKernelPass{});
#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
  FPM.addPass(PHIsToAllocasPass{});
#endif
  FPM.addPass(llvm::LoopSimplifyPass{});

  FPM.addPass(CanonicalizeBarriersPass{});
  if (IsSscp)
    FPM.addPass(KernelFlatteningPass{});
  FPM.addPass(SubCfgFormationPass{IsSscp});
  FPM.addPass(RemoveBarrierCallsPass{});

  if (Opt == OptLevel::O3)
    FPM.addPass(KernelFlatteningPass{});
  if (Opt != OptLevel::O0)
    FPM.addPass(LoopsParallelMarkerPass{});
  
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
}

} // namespace hipsycl::compiler
