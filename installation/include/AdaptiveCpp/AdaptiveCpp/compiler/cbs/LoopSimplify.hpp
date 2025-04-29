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
#ifndef HIPSYCL_LOOPSIMPLIFY_HPP
#define HIPSYCL_LOOPSIMPLIFY_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace hipsycl {
namespace compiler {

// Wrapper pass for llvm::simplifyLoop that is only run on nd-range kernels.
class LoopSimplifyPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit LoopSimplifyPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL loop simplify pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_LOOPSIMPLIFY_HPP
