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
#include "hipSYCL/compiler/llvm-to-backend/clspv/RemoveUnusedIntrinsicsPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

namespace hipsycl {
namespace compiler {

// clspv ignores freeze, assume, and lifetime intrinsics. Remove them early
// to make the IR we output cleaner to read and debug.
llvm::PreservedAnalyses
RemoveUnusedIntrinsicsPass::run(llvm::Function &F,
                                llvm::FunctionAnalysisManager &) {
  llvm::SmallVector<llvm::Instruction *> IToRemove;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *FI = llvm::dyn_cast<llvm::FreezeInst>(&I)) {
        FI->replaceAllUsesWith(FI->getOperand(0));
        FI->dropAllReferences();
        IToRemove.push_back(FI);
      } else if (llvm::CallBase *CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
        auto *CalledF = CB->getCalledFunction();
        assert(CalledF);
        switch (CalledF->getIntrinsicID()) {
        case llvm::Intrinsic::assume:
        case llvm::Intrinsic::lifetime_start:
        case llvm::Intrinsic::lifetime_end:
          CB->replaceAllUsesWith(llvm::UndefValue::get(CB->getType()));
          IToRemove.push_back(CB);
          break;
        default:
          break;
        }
      }
    }
  }

  for (auto *I : IToRemove) {
    I->eraseFromParent();
  }

  return IToRemove.empty() ? llvm::PreservedAnalyses::all()
                           : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
