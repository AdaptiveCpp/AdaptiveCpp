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
#ifndef HIPSYCL_REMOVEBARRIERCALLS_HPP
#define HIPSYCL_REMOVEBARRIERCALLS_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace hipsycl {
namespace compiler {

// prunes barrier calls at the end of the pipeline
class RemoveBarrierCallsPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit RemoveBarrierCallsPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL barrier call removal pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

class RemoveBarrierCallsPass : public llvm::PassInfoMixin<RemoveBarrierCallsPass> {
public:
  explicit RemoveBarrierCallsPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_REMOVEBARRIERCALLS_HPP
