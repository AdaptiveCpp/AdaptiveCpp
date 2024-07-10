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
#include "hipSYCL/compiler/cbs/SimplifyKernel.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/MemorySSAUpdater.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Transforms/Utils/LoopRotationUtils.h>
#include <llvm/Transforms/Utils/LoopUtils.h>

namespace {
using namespace hipsycl::compiler;
bool simplifyKernel(llvm::Function &F, llvm::DominatorTree &DT, llvm::AssumptionCache &AC) {
  bool Changed = true;
  HIPSYCL_DEBUG_INFO << "Promote allocas in " << F.getName() << "\n";
  utils::promoteAllocas(&F.getEntryBlock(), DT, AC);
  return Changed;
}
} // namespace

void hipsycl::compiler::SimplifyKernelPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<llvm::AssumptionCacheTracker>();
  AU.addPreserved<llvm::AssumptionCacheTracker>();
}

bool hipsycl::compiler::SimplifyKernelPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<llvm::AssumptionCacheTracker>().getAssumptionCache(F);
  return simplifyKernel(F, DT, AC);
}

llvm::PreservedAnalyses
hipsycl::compiler::SimplifyKernelPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  const auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAMProxy.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  assert(SAA && "Must have SplitterAnnotationAnalysis cached!");
  if (!SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<llvm::AssumptionAnalysis>(F);

  if (!simplifyKernel(F, DT, AC))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserveSet<llvm::CFGAnalyses>();
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}

char hipsycl::compiler::SimplifyKernelPassLegacy::ID = 0;
