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
#include "hipSYCL/compiler/llvm-to-backend/clspv/PtrToIntPass.hpp"

#include <llvm/IR/Instructions.h>

namespace hipsycl {
namespace compiler {

// Pass for fixing SROA-d components from large structs so that we don't
// lose the address space of pointers between PtrToInt then IntToPtr.
llvm::PreservedAnalyses PtrToIntPass::run(llvm::Function &F,
                                          llvm::FunctionAnalysisManager &) {
  llvm::SmallVector<llvm::Instruction *> InstToRemove;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *PTI = llvm::dyn_cast<llvm::PtrToIntInst>(&I)) {
        if (PTI->getPointerAddressSpace() == 1) {
          llvm::Value *Op = PTI->getPointerOperand();
          for (auto U : PTI->users()) {
            if (auto ITP = llvm::dyn_cast<llvm::IntToPtrInst>(U)) {
              ITP->replaceAllUsesWith(Op);
              InstToRemove.push_back(ITP);
            }
          }
        }
      }
    }
  }

  for (auto *I : InstToRemove) {
    I->eraseFromParent();
  }

  return InstToRemove.empty() ? llvm::PreservedAnalyses::all()
                              : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
