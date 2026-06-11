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
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

// GEP instructions may now have an address space that isn't reflected in its
// users, recreate the GEP so that it has the correct address space and replace
// uses.
bool fixupGEP(llvm::Function &F) {
  std::vector<llvm::Instruction *> InstToDel;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {
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

  return !InstToDel.empty();
}

// Need to recreate builtin declaration with address spaces where previously
// there was no address space on operands.
bool fixupMemInstrinsic(llvm::Function &F) {
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
      }
    }
  }
  for (auto I : InstToDel) {
    I->eraseFromParent();
  }

  return !InstToDel.empty();
}

// Fix icmp instructions comparing a pointer with nonzero address
// space in the first operand against a nullptr with no address space.
// Transformed by giving the nullptr constant the same address space.
bool fixupICMPNull(llvm::Function &F) {
  bool DidTransform = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto ICmp = llvm::dyn_cast<llvm::ICmpInst>(&I)) {
        auto Op0 = ICmp->getOperand(0);
        auto Op1 = ICmp->getOperand(1);
        auto Op0PtrType = llvm::dyn_cast<llvm::PointerType>(Op0->getType());
        auto Op1PtrType = llvm::dyn_cast<llvm::PointerType>(Op1->getType());
        if (Op0PtrType && Op1PtrType) {
          if (Op0PtrType->getAddressSpace() != 0 &&
              0 == Op1PtrType->getAddressSpace()) {
            if (auto Op1Const = llvm::dyn_cast<llvm::Constant>(Op1);
                Op1Const && Op1Const->isNullValue()) {
              auto newNull = llvm::Constant::getNullValue(Op0PtrType);
              ICmp->setOperand(1, newNull);
              DidTransform = true;
            }
          }
        }
      }
    }
  }
  return DidTransform;
}

// Remove address space cast instructions and replace uses with pointer operand
bool removeCasts(llvm::Function &F) {
  llvm::SmallVector<llvm::Instruction *> ASCToRemove;
  bool DidTransform = false;

  auto processConstExpr = [&DidTransform](llvm::Value *Val,
                                          llvm::ConstantExpr *CE) {
    auto CEI = CE->getAsInstruction();
    if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(CEI)) {
      llvm::Value *ASCop = ASC->getPointerOperand();
      Val->replaceAllUsesWith(ASCop);
      DidTransform = true;
    }
    CEI->deleteValue(); // Don't leak instruction without parent
  };

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
          processConstExpr(Op, CE);
        }
      } else if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(&I)) {
        auto *Op = Store->getValueOperand();
        if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(Op)) {
          processConstExpr(Op, CE);
        }
        auto *Ptr = Store->getPointerOperand();
        if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(Ptr)) {
          processConstExpr(Ptr, CE);
        }
      } else if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        auto *Ptr = Load->getPointerOperand();
        if (auto CE = llvm::dyn_cast<llvm::ConstantExpr>(Ptr)) {
          processConstExpr(Ptr, CE);
        }
      }
    }
  }

  for (auto *I : ASCToRemove) {
    I->eraseFromParent();
  }

  return (!ASCToRemove.empty() || DidTransform);
}

// For alloca instructions which are created from SROA and hold a pointer, make
// sure that the pointer type retains the pointer address space. This is inferred
// from the store users which are used to create a new alloca with the correct type.
bool fixupAllocas(llvm::Function &F) {
  llvm::SmallVector<llvm::Instruction *> InstsToDel;
  // Loop over Store instructions, and replace any allocas that don't match
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto Store = llvm::dyn_cast<llvm::StoreInst>(&I)) {
        auto Alloca =
            llvm::dyn_cast<llvm::AllocaInst>(Store->getPointerOperand());
        if (!Alloca) {
          continue;
        }

        auto StoreVal = Store->getValueOperand();
        auto StoreType = llvm::dyn_cast<llvm::PointerType>(StoreVal->getType());
        auto AllocPtrType =
            llvm::dyn_cast<llvm::PointerType>(Alloca->getAllocatedType());
        if (!StoreType || !AllocPtrType) {
          continue;
        }

        if (StoreType->getAddressSpace() != AllocPtrType->getAddressSpace()) {
          llvm::IRBuilder Builder(Alloca);
          auto NewAlloca = Builder.CreateAlloca(StoreType);
          Alloca->replaceAllUsesWith(NewAlloca);
          InstsToDel.push_back(Alloca);
        }
      }
    }
  }

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto Load = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        if (auto Alloca =
                llvm::dyn_cast<llvm::AllocaInst>(Load->getPointerOperand())) {
          if (auto AllocType = llvm::dyn_cast<llvm::PointerType>(
                  Alloca->getAllocatedType())) {
            if (AllocType->getAddressSpace() !=
                Load->getType()->getPointerAddressSpace()) {
              llvm::IRBuilder Builder(Load);
              // Don't retain any volatile attribute to enable later mem2reg
              // optimizations
              auto NewLoad = Builder.CreateAlignedLoad(AllocType, Alloca,
                                                       Load->getAlign());
              Load->replaceAllUsesWith(NewLoad);
              InstsToDel.push_back(Load);
            }
          }
        }
      }
    }
  }

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *ASC = llvm::dyn_cast<llvm::AddrSpaceCastInst>(&I)) {
        if (ASC->getDestAddressSpace() == ASC->getSrcAddressSpace()) {
          ASC->replaceAllUsesWith(ASC->getPointerOperand());
          InstsToDel.push_back(ASC);
        }
      }
    }
  }

  for (auto *I : InstsToDel) {
    I->eraseFromParent();
  }

  return !InstsToDel.empty();
}

llvm::PreservedAnalyses
AddrSpaceCastRemovalPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
  // Remove casting away the address space from arguments, as otherwise
  // clspv doesn't know how to lower the generic pointer to SPIRV.
  // Done by remove the address space cast instructions and replacing all uses
  // with the instructions pointer operand. Some uses then need to be fixed up
  // to be correctly formed.
  bool DidTransform = removeCasts(F);
  DidTransform |= fixupICMPNull(F);
  DidTransform |= fixupGEP(F);
  DidTransform |= fixupMemInstrinsic(F);
  DidTransform |= fixupAllocas(F);
  return DidTransform ? llvm::PreservedAnalyses::all()
                      : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
