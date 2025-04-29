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
#ifndef HIPSYCL_DEAD_ARGUMENT_ELIMINATION_PASS_HPP
#define HIPSYCL_DEAD_ARGUMENT_ELIMINATION_PASS_HPP

#include <unordered_map>
#include <functional>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>


namespace hipsycl {
namespace compiler {

class DeadArgumentEliminationPass : public llvm::PassInfoMixin<DeadArgumentEliminationPass> {
public:
  DeadArgumentEliminationPass(llvm::Function *TargetFunction,
                              llvm::SmallVector<int> *RetainedArgumentIndicesOut = nullptr,
                              std::function<void(llvm::Function *Old, llvm::Function *New)>
                                  *ReplacementFunctionAttributeTransfer = nullptr);

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  llvm::Function* TargetFunction;
  llvm::SmallVector<int>* RetainedArguments;
  std::function<void(llvm::Function*, llvm::Function*)>* ReplacementFunctionAttributeTransfer;
};



}
}

#endif

