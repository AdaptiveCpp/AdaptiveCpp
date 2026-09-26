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
#ifndef HIPSYCL_SSCP_INTEGER_LEGALIZATION_PASS_HPP
#define HIPSYCL_SSCP_INTEGER_LEGALIZATION_PASS_HPP

#include <llvm/IR/PassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/ADT/DenseMap.h>

#include <optional>
#include <string>

namespace hipsycl {
namespace compiler {

class IntegerLegalizationPass
  : public llvm::PassInfoMixin<IntegerLegalizationPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);
  std::optional<std::string> getErrorMessage() const { return ErrorMessage; }

private:
  llvm::Value* getPromoted(llvm::Value* V);
  llvm::Value* promoteResult(llvm::IRBuilder<> &B, llvm::Value* V);
  bool rebuildLegal(llvm::IRBuilder<> &B, llvm::Instruction* I);
  llvm::PreservedAnalyses fail(llvm::Instruction* I);

  llvm::DenseMap<llvm::Value *, llvm::Value *> Promoted;
  llvm::Module* M = nullptr;
  std::optional<std::string> ErrorMessage = std::nullopt;
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_INTEGER_LEGALIZATION_PASS_HPP