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
#include "hipSYCL/compiler/llvm-to-backend/clspv/AddrSpaceRestorePass.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>

namespace hipsycl {
namespace compiler {

// Pass for fixing instructions that have lost address space during
// optimization
llvm::PreservedAnalyses
AddrSpaceRestorePass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  std::vector<llvm::Instruction *> InstToDel;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto MemCpy = llvm::dyn_cast<llvm::MemCpyInst>(&I)) {
        llvm::IRBuilder Builder(MemCpy);
        auto newCI =
            Builder.CreateMemCpy(MemCpy->getRawDest(), MemCpy->getDestAlign(),
                                 MemCpy->getRawSource(),
                                 MemCpy->getSourceAlign(), MemCpy->getLength());
        newCI->setTailCall(MemCpy->isTailCall());
        MemCpy->replaceAllUsesWith(newCI);
        InstToDel.push_back(MemCpy);
      } else if (auto MemSet = llvm::dyn_cast<llvm::MemSetInst>(&I)) {
        llvm::IRBuilder Builder(MemSet);
        auto newCI =
            Builder.CreateMemSet(MemSet->getRawDest(), MemSet->getValue(),
                                 MemSet->getLength(), MemSet->getDestAlign());
        newCI->setTailCall(MemSet->isTailCall());
        MemSet->replaceAllUsesWith(newCI);
        InstToDel.push_back(MemSet);
      } else if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {
        unsigned AddrSpace = GEP->getAddressSpace();
        if (0 != AddrSpace) {
          std::vector<llvm::Value *> indices;
          for (auto &i : GEP->indices()) {
            indices.push_back(llvm::cast<llvm::Value>(&i));
          }

          llvm::IRBuilder Builder(GEP);
          auto newGEP = Builder.CreateGEP(GEP->getSourceElementType(),
                                          GEP->getPointerOperand(), indices);
          InstToDel.push_back(GEP);
          GEP->replaceAllUsesWith(newGEP);
        }
      }
    }
  }
  for (auto I : InstToDel) {
    I->eraseFromParent();
  }

  return InstToDel.empty() ? llvm::PreservedAnalyses::all()
                           : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
