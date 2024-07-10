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
#ifndef HIPSYCL_PHISTOALLOCAS_HPP
#define HIPSYCL_PHISTOALLOCAS_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace hipsycl {
namespace compiler {

// demotes non-wi-loop phis to allocas - only used if HIPSYCL_NO_PHIS_IN_SPLIT is set.
class PHIsToAllocasPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit PHIsToAllocasPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL PHI to alloca demote pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

class PHIsToAllocasPass : public llvm::PassInfoMixin<PHIsToAllocasPass> {
public:
  explicit PHIsToAllocasPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_PHISTOALLOCAS_HPP
