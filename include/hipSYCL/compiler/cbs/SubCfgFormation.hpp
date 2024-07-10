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
#ifndef HIPSYCL_SUBCFGFORMATION_HPP
#define HIPSYCL_SUBCFGFORMATION_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace hipsycl {
namespace compiler {

constexpr size_t EntryBarrierId = 0;
constexpr size_t ExitBarrierId = -1;

// performs the main CBS transformation
class SubCfgFormationPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit SubCfgFormationPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL sub-CFG formation pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

class SubCfgFormationPass : public llvm::PassInfoMixin<SubCfgFormationPass> {
  bool IsSscp_;
public:
  explicit SubCfgFormationPass(bool IsSscp) : IsSscp_(IsSscp) {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SUBCFGFORMATION_HPP
