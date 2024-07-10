/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
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

#if (LLVM_VERSION_MAJOR < 16) || defined(IS_ROCM_CLANG_VERSION_5_5_0)
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
