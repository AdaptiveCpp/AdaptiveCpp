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

// Fix atomic builtin calls that take an opaque ptr without an address
// space but pointer value for used for builtin parameter has an addrspace. Done
// by creating an address space cast for the relevant operand.
bool fixupAtomicBuiltins(llvm::Function &F) {
  static const llvm::SmallVector<llvm::StringRef> AtomicFuncs = {
      "_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope",
      "_Z20atomic_load_explicitPU3AS4VU7_Atomicl12memory_order12memory_scope",
      "_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z21atomic_store_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",
      "_Z24atomic_exchange_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z24atomic_exchange_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",

      "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
      "scope",

      "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
      "scope",

      "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
      "scope",

      "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicjj12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicmm12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicff12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicdd12memory_order12memory_"
      "scope",

      "_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",

      "_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",

      "_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicii12memory_order12memory_"
      "scope",
      "_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicll12memory_order12memory_"
      "scope",

      "_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_"
      "AtomiciPU3AS4ii12memory_orderS4_12memory_scope",
      "_Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_"
      "AtomiciPU3AS4jj12memory_orderS4_12memory_scope",

      "_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_"
      "AtomiciPU3AS4ii12memory_orderS4_12memory_scope",
      "_Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_"
      "AtomiciPU3AS4jj12memory_orderS4_12memory_scope",
  };

  bool DidTransform = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
        auto Func = CI->getCalledFunction();
        auto Name = Func->getName();
        if (std::find(AtomicFuncs.begin(), AtomicFuncs.end(), Name) !=
            AtomicFuncs.end()) {
          auto Arg0 = CI->getArgOperand(0);
          auto ArgType = Arg0->getType();
          if (ArgType->getPointerAddressSpace() != 0) {
            llvm::IRBuilder Builder(CI);
            llvm::PointerType *GenType =
                llvm::PointerType::get(F.getParent()->getContext(), 0);
            auto AddrSpaceCast = Builder.CreateAddrSpaceCast(Arg0, GenType);
            CI->setArgOperand(0, AddrSpaceCast);
            DidTransform = true;
          }
        }
      }
    }
  }
  return DidTransform;
}

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
          CEI->deleteValue(); // Don't leak instruction without parent
        }
      }
    }
  }

  for (auto *I : ASCToRemove) {
    I->eraseFromParent();
  }

  return (ASCToRemove.empty() || !DidGEPTransform);
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
  DidTransform |= fixupAtomicBuiltins(F);
  return DidTransform ? llvm::PreservedAnalyses::all()
                      : llvm::PreservedAnalyses::none();
}
} // namespace compiler
} // namespace hipsycl
