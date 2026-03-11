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
#pragma once

#include <llvm/IR/PassManager.h>

namespace llvm {
class AllocaInst;
class StructType;
} // namespace llvm

namespace hipsycl {
namespace compiler {

class SROAParallelForPass : public llvm::PassInfoMixin<SROAParallelForPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);

private:
  void findParallelForStruct(llvm::Function &F);

  llvm::AllocaInst *MAllocaToRemove = nullptr;
  llvm::StructType *MArgsStruct = nullptr;
};

} // namespace compiler
} // namespace hipsycl
