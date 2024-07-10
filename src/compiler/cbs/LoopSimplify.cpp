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
#include "hipSYCL/compiler/cbs/LoopSimplify.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Dominators.h>

namespace hipsycl::compiler {
char LoopSimplifyPassLegacy::ID = 0;

void LoopSimplifyPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::ScalarEvolutionWrapperPass>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool LoopSimplifyPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  auto *SCEWP = getAnalysisIfAvailable<llvm::ScalarEvolutionWrapperPass>();
  auto *SE = SCEWP ? &SCEWP->getSE() : nullptr;

  bool Changed = false;
  for (auto *L : LI) {
    HIPSYCL_DEBUG_INFO << "[LoopSimplify] Simplifying loop: " << L->getHeader()->getName() << "\n";
    Changed |= llvm::simplifyLoop(L, &DT, &LI, SE, nullptr, nullptr, false);
  }
  return Changed;
}
} // namespace hipsycl::compiler
