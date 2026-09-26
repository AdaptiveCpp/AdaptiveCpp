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

namespace llvm {
  class Value;
} // namespace llvm

namespace hipsycl {
namespace compiler {

class IntegerLegalizationPass
  : public llvm::PassInfoMixin<IntegerLegalizationPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
  llvm::Value* getPromoted(llvm::Value* V);

  llvm::DenseMap<llvm::Value *, llvm::Value *> Promoted;
  llvm::Module* M = nullptr;
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_INTEGER_LEGALIZATION_PASS_HPP