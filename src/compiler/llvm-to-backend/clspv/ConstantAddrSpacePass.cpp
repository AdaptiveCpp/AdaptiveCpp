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
#include "hipSYCL/compiler/llvm-to-backend/clspv/ConstantAddrSpacePass.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

void replaceUsers(llvm::GlobalVariable *OldGlobal,
                  llvm::GlobalVariable *NewGlobal) {
  std::vector<llvm::Instruction *> InstToDel;
  for (auto User : OldGlobal->users()) {
    if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(User)) {
      std::vector<llvm::Value *> indices;
      for (auto &i : GEP->indices()) {
        indices.push_back(llvm::cast<llvm::Value>(&i));
      }

      llvm::IRBuilder Builder(GEP);
      auto newGEP =
          Builder.CreateGEP(GEP->getSourceElementType(), NewGlobal, indices);
      GEP->replaceAllUsesWith(newGEP);
      InstToDel.push_back(GEP);
    } else if (auto MemCpy = llvm::dyn_cast<llvm::MemCpyInst>(User)) {
      llvm::IRBuilder Builder(MemCpy);
      llvm::Value *newSource =
          MemCpy->getSource() == OldGlobal ? NewGlobal : MemCpy->getSource();
      llvm::Value *newDest =
          MemCpy->getDest() == OldGlobal ? NewGlobal : MemCpy->getDest();
      auto newCI =
          Builder.CreateMemCpy(newDest, MemCpy->getDestAlign(), newSource,
                               MemCpy->getSourceAlign(), MemCpy->getLength());
      newCI->setTailCall(MemCpy->isTailCall());
      MemCpy->replaceAllUsesWith(newCI);
      InstToDel.push_back(MemCpy);
    } else if (auto MemSet = llvm::dyn_cast<llvm::MemSetInst>(User)) {
      llvm::IRBuilder Builder(MemSet);
      auto newCI =
          Builder.CreateMemSet(NewGlobal, MemSet->getValue(),
                               MemSet->getLength(), MemSet->getDestAlign());
      newCI->setTailCall(MemSet->isTailCall());
      MemSet->replaceAllUsesWith(newCI);
      InstToDel.push_back(MemSet);
    } else {
      llvm_unreachable("Unsupported user");
    }
  }

  for (auto *I : InstToDel) {
    I->eraseFromParent();
  }
}

// Ensure const global variables use the constant address space (2).
llvm::PreservedAnalyses
ConstantAddrSpacePass::run(llvm::Module &M, llvm::ModuleAnalysisManager &) {

  llvm::SmallVector<llvm::GlobalVariable *> GlobalsToRemove;
  for (auto &Global : M.globals()) {
    if (!Global.isConstant()) {
      continue;
    }
    if (auto GlobalType = llvm::dyn_cast<llvm::PointerType>(Global.getType());
        GlobalType && GlobalType->getAddressSpace() == 0) {
      auto InitType = Global.getInitializer()->getType();
      llvm::GlobalVariable *NewGlobal = new llvm::GlobalVariable(
          M, InitType, true, Global.getLinkage(), Global.getInitializer(),
          Global.getName(), &Global, llvm::GlobalValue::NotThreadLocal,
          2 /* new address space */);
      NewGlobal->setAlignment(Global.getAlign());
      replaceUsers(&Global, NewGlobal);
      GlobalsToRemove.push_back(&Global);
    }
  }

  for (auto *G : GlobalsToRemove) {
    G->eraseFromParent();
  }

  return GlobalsToRemove.empty() ? llvm::PreservedAnalyses::all()
                                 : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
