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
#ifndef HIPSYCL_KERNELFLATTENING_HPP
#define HIPSYCL_KERNELFLATTENING_HPP

#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>

namespace hipsycl {
namespace compiler {
class KernelFlatteningPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit KernelFlatteningPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL kernel flattening pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

class KernelFlatteningPass : public llvm::PassInfoMixin<KernelFlatteningPass> {
public:
  explicit KernelFlatteningPass() {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return false; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_KERNELFLATTENING_HPP
