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
#ifndef HIPSYCL_LOOPSPLITTERINLINING_HPP
#define HIPSYCL_LOOPSPLITTERINLINING_HPP

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace hipsycl {
namespace compiler {

// inlines the call trees of the barriers and replaces them by a builtin
class LoopSplitterInliningPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit LoopSplitterInliningPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL loop splitter inlining pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &L) override;
};

class LoopSplitterInliningPass : public llvm::PassInfoMixin<LoopSplitterInliningPass> {
public:
  explicit LoopSplitterInliningPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_LOOPSPLITTERINLINING_HPP
