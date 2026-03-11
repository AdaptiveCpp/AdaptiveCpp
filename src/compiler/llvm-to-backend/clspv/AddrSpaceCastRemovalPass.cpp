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
#include "hipSYCL/compiler/llvm-to-backend/clspv/AddrSpaceCastRemovalPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

namespace hipsycl {
namespace compiler {

// Remove casting away the address space from arguments, as otherwise
// clspv doesn't know how to lower the generic pointer to SPIRV
llvm::PreservedAnalyses
AddrSpaceCastRemovalPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
  llvm::SmallVector<llvm::Instruction *> ASCToRemove;
  bool DidGEPTransform = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(&I)) {
        if (ASC->getDestAddressSpace() == 0) {
          llvm::Value *Op = ASC->getPointerOperand();
          ASC->replaceAllUsesWith(Op);
          ASCToRemove.push_back(ASC);
        }
      } else if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {
        auto *Op = GEP->getPointerOperand();
        if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(Op)) {
          auto CEI = CE->getAsInstruction();
          if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(CEI)) {
            llvm::Value *ASCop = ASC->getPointerOperand();
            Op->replaceAllUsesWith(ASCop);
            DidGEPTransform = true;
          }
        }
      }
    }
  }

  for (auto *I : ASCToRemove) {
    I->eraseFromParent();
  }

  return (ASCToRemove.empty() || !DidGEPTransform)
             ? llvm::PreservedAnalyses::all()
             : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
