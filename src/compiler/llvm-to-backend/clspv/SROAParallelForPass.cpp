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
#include "hipSYCL/compiler/llvm-to-backend/clspv/SROAParallelForPass.hpp"

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

void SROAParallelForPass::findParallelForStruct(llvm::Function &F) {
  // Find Alloca and args struct
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        llvm::Type *AllocaTy = AI->getAllocatedType();
        if (llvm::StructType *StructT =
                llvm::dyn_cast<llvm::StructType>(AllocaTy)) {
          if (StructT->getName().starts_with(
                  "class.hipsycl::glue::__sscp_dispatch::basic_parallel_"
                  "for")) {
            assert(StructT->getNumElements() == 2);
            MArgsStruct =
                llvm::cast<llvm::StructType>(StructT->getElementType(0));
            MAllocaToRemove = AI;
          }
        }
      }
    }
  }
}

// if SROA hasn't broken down the basic_parallel_for struc then we need to
// manually do it so that clspv can understand what's going on.
llvm::PreservedAnalyses
SROAParallelForPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  findParallelForStruct(F);
  if (!MAllocaToRemove) {
    return llvm::PreservedAnalyses::all();
  }

  llvm::IRBuilder Builder(MAllocaToRemove);
  llvm::SmallVector<llvm::Instruction *> InstsToRemove;
  llvm::SmallVector<llvm::AllocaInst *> NewAllocs;

  for (llvm::Type *Ty : MArgsStruct->elements()) {
    llvm::AllocaInst *AI = Builder.CreateAlloca(Ty);
    AI->setAlignment(MAllocaToRemove->getAlign());
    NewAllocs.push_back(AI);
  }

  for (auto U : MAllocaToRemove->users()) {
    if (auto Store = llvm::dyn_cast<llvm::StoreInst>(U)) {
      assert(Store->getPointerOperand() == MAllocaToRemove);
      Builder.SetInsertPoint(Store);
      auto NewStore =
          Builder.CreateStore(Store->getValueOperand(), NewAllocs[0]);
      NewStore->setAlignment(Store->getAlign());
      InstsToRemove.push_back(Store);
    }

    if (auto Load = llvm::dyn_cast<llvm::LoadInst>(U)) {
      assert(Load->getPointerOperand() == MAllocaToRemove);
      Builder.SetInsertPoint(Load);
      llvm::AllocaInst *NewAlloc = NewAllocs[0];
      auto NewLoad = Builder.CreateLoad(NewAlloc->getType(), NewAlloc);
      NewLoad->setAlignment(Load->getAlign());
      Load->replaceAllUsesWith(NewLoad);
      InstsToRemove.push_back(Load);
    }

    if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
      assert(GEP->getPointerOperand() == MAllocaToRemove);
      Builder.SetInsertPoint(GEP);

      std::vector<llvm::Value *> Indices;
      for (auto &Idx : GEP->indices()) {
        Indices.push_back(llvm::cast<llvm::Value>(&Idx));
      }
      llvm::ConstantInt *Idx0 = llvm::cast<llvm::ConstantInt>(Indices[0]);
      unsigned I0 = Idx0->getValue().getZExtValue();

      unsigned Offset = 0;
      unsigned OldOffset = 0;
      llvm::AllocaInst *NewAlloca = nullptr;
      for (auto A : NewAllocs) {
        if (Offset >= I0) {
          OldOffset = Offset;
          NewAlloca = A;
          break;
        }
        OldOffset = Offset;
        NewAlloca = A;
        Offset += *A->getAllocationSize(F.getParent()->getDataLayout());
      }
      I0 -= OldOffset;

      Indices[0] = Builder.getIntN(Idx0->getIntegerType()->getBitWidth(), I0);
      auto NewGEP = Builder.CreateInBoundsGEP(GEP->getResultElementType(),
                                              NewAlloca, Indices);
      GEP->replaceAllUsesWith(NewGEP);
      InstsToRemove.push_back(GEP);
    }
  }

  for (auto *I : InstsToRemove) {
    I->eraseFromParent();
  }
  MAllocaToRemove->eraseFromParent();

  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
