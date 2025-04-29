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
#ifndef HIPSYCL_SIMPLIFYKERNEL_HPP
#define HIPSYCL_SIMPLIFYKERNEL_HPP

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace hipsycl {
namespace compiler {

// wrapper pass for llvm::PromoteMemToReg that only runs on nd-range kernels
class SimplifyKernelPassLegacy : public llvm::FunctionPass {

public:
  static char ID;

  explicit SimplifyKernelPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL kernel simplify pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &L) override;
};

class SimplifyKernelPass : public llvm::PassInfoMixin<SimplifyKernelPass> {

public:
  explicit SimplifyKernelPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SIMPLIFYKERNEL_HPP
