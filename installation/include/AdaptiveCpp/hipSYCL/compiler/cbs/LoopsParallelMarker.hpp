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
#ifndef HIPSYCL_LOOPSPARALLELMARKER_HPP
#define HIPSYCL_LOOPSPARALLELMARKER_HPP

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace hipsycl {
namespace compiler {

// marks the wi-loops as parallel (vectorizable) and enables vectorization.
class LoopsParallelMarkerPassLegacy : public llvm::FunctionPass {

public:
  static char ID;

  explicit LoopsParallelMarkerPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL loop parallel marking pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &L) override;
};

class LoopsParallelMarkerPass : public llvm::PassInfoMixin<LoopsParallelMarkerPass> {

public:
  explicit LoopsParallelMarkerPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return false; }
};
} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_LOOPSPARALLELMARKER_HPP
