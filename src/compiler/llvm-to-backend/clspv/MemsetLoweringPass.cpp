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
#include "hipSYCL/compiler/llvm-to-backend/clspv/MemsetLoweringPass.hpp"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

// Convert memset byte into repeated pattern:
//
// memset(..., 0xAB, ...)
//
// i32 value => 0xABABABAB
llvm::Value *buildSplatValue(llvm::IRBuilder<> &B, llvm::Value *ByteVal,
                             llvm::Type *ElemTy) {
  llvm::LLVMContext &Ctx = B.getContext();
  if (!ElemTy->isIntegerTy()) {
    llvm_unreachable("Expected Integer Type");
  }

  auto *ByteConstI = llvm::dyn_cast<llvm::ConstantInt>(ByteVal);
  if (!ByteConstI) {
    llvm_unreachable("Expected Constant pattern");
  }

  uint64_t ByteConst = ByteConstI->getZExtValue();

  unsigned Bits = ElemTy->getIntegerBitWidth();
  llvm::APInt Pattern(Bits, 0);
  for (unsigned i = 0; i < Bits; i += 8) {
    Pattern |= llvm::APInt(Bits, ByteConst) << i;
  }

  return llvm::ConstantInt::get(Ctx, Pattern);
}

llvm::Type *inferDataType(llvm::MemSetInst *MS) {
  llvm::LLVMContext &Ctx = MS->getContext();
  llvm::Value *Dst = MS->getDest();
  llvm::Type *ElemTy = nullptr;
  if (auto GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Dst)) {
    ElemTy = GEP->getResultElementType();
  } else {
    auto Align = MS->getDestAlign();
    if (!Align) {
      llvm_unreachable("Need alignment to work out type");
    }
    const uint64_t AlignVal = Align->value();

    if (AlignVal % sizeof(uint32_t) == 0) {
      ElemTy = llvm::Type::getInt32Ty(Ctx);
    } else if (AlignVal % sizeof(uint16_t) == 0) {
      ElemTy = llvm::Type::Type::getInt16Ty(Ctx);
    } else {
      ElemTy = llvm::Type::getInt8Ty(Ctx);
    }
  }
  return ElemTy;
}

void lowerMemsetStaticLen(llvm::MemSetInst *MS) {
  // Work out the data type to store based on dst pointer
  llvm::Value *Dst = MS->getDest();
  llvm::Type *ElemTy = inferDataType(MS);

  auto M = MS->getModule();
  const llvm::DataLayout &DL = MS->getModule()->getDataLayout();
  uint64_t ElemSize = DL.getTypeStoreSize(ElemTy);

  llvm::IRBuilder<> Builder(MS);
  auto Val = MS->getValue();
  llvm::Value *StoreVal = buildSplatValue(Builder, Val, ElemTy);

  auto NumBytes =
      llvm::cast<llvm::ConstantInt>(MS->getLength())->getZExtValue();
  const auto NumStores = NumBytes / ElemSize;
  assert((NumBytes == NumStores * ElemSize) &&
         "Null memset can't be divided evenly across multiple stores.");

  auto I32Ty = llvm::Type::getInt32Ty(M->getContext());
  for (uint32_t i = 0; i < NumStores; i++) {
    auto Index = llvm::ConstantInt::get(I32Ty, i);
    llvm::Value *Ptr = Builder.CreateInBoundsGEP(ElemTy, Dst, Index);
    Builder.CreateStore(StoreVal, Ptr);
  }

  // Remove original memset intrinsic
  MS->eraseFromParent();
}

void lowerMemsetDynamicLen(llvm::MemSetInst *MS) {
  // Create a loop to perform GEP and Stores, since the length of the memset is
  // dynamic we don't know the number of loop iterations to manually unroll the
  // loop.
  llvm::LLVMContext &Ctx = MS->getContext();
  llvm::Function *F = MS->getFunction();
  llvm::BasicBlock *OrigBB = MS->getParent();

  // Split block at memset
  llvm::BasicBlock *AfterBB =
      OrigBB->splitBasicBlock(MS->getIterator(), "memset.after");

  // Remove unconditional branch created by split
  OrigBB->getTerminator()->eraseFromParent();

  // Create loop blocks
  llvm::BasicBlock *LoopBB =
      llvm::BasicBlock::Create(Ctx, "memset.loop", F, AfterBB);
  llvm::BasicBlock *BodyBB =
      llvm::BasicBlock::Create(Ctx, "memset.body", F, AfterBB);

  // Branch into loop
  llvm::IRBuilder<> Builder(OrigBB);
  Builder.CreateBr(LoopBB);

  // Loop header BB
  Builder.SetInsertPoint(LoopBB);

  // Define Phi
  llvm::Type *Int64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::PHINode *Index = Builder.CreatePHI(Int64Ty, 2, "i");
  Index->addIncoming(llvm::ConstantInt::get(Int64Ty, 0), OrigBB);

  // Work out the data type to store based on dst pointer
  llvm::Value *Dst = MS->getDest();
  llvm::Type *ElemTy = inferDataType(MS);

  // Iterations is the byte operand to memset divided by size of the type
  // Warning: This assumes no remainder.
  llvm::Value *Len = MS->getLength();
  const llvm::DataLayout &DL = F->getParent()->getDataLayout();
  uint64_t ElemSize = DL.getTypeStoreSize(ElemTy);
  llvm::Value *Iters =
      Builder.CreateUDiv(Len, llvm::ConstantInt::get(Int64Ty, ElemSize));

  // Conditional to branch to completion BB
  llvm::Value *Done = Builder.CreateICmpEQ(Index, Iters);
  Builder.CreateCondBr(Done, AfterBB, BodyBB);

  // Create body basic block
  Builder.SetInsertPoint(BodyBB);

  // Create GEP into destination pointer with loop index
  llvm::Value *Ptr = Builder.CreateInBoundsGEP(ElemTy, Dst, Index);

  // Create store into GEP with memset byte value splatted across GEP data type
  auto Val = MS->getValue();
  llvm::Value *StoreVal = buildSplatValue(Builder, Val, ElemTy);
  Builder.CreateStore(StoreVal, Ptr);

  // Increment loop counter
  llvm::Value *Next =
      Builder.CreateAdd(Index, llvm::ConstantInt::get(Int64Ty, 1));

  Builder.CreateBr(LoopBB);
  Index->addIncoming(Next, BodyBB);

  // Remove original memset intrinsic
  MS->eraseFromParent();
}

llvm::PreservedAnalyses
MemsetLoweringPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  llvm::SmallVector<llvm::MemSetInst *, 8> Memsets;

  // Collect memset intrinsics first
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      if (auto *MS = llvm::dyn_cast<llvm::MemSetInst>(&I)) {
        // We only support constant initializers in this pass
        if (auto Initializer =
                llvm::dyn_cast<llvm::ConstantInt>(MS->getValue())) {
          // clspv can handle memsets where the initializer is a constant int
          // that is 0, and the length is known at compile time. In such cases
          // leave the memset untouched.
          if (0 != Initializer->getZExtValue() ||
              !llvm::isa<llvm::ConstantInt>(MS->getLength())) {
            Memsets.push_back(MS);
          }
        }
      }
    }
  }

  if (Memsets.empty()) {
    return llvm::PreservedAnalyses::all();
  }

  for (llvm::MemSetInst *MS : Memsets) {
    if (llvm::isa<llvm::ConstantInt>(MS->getLength())) {
      lowerMemsetStaticLen(MS);
    } else {
      lowerMemsetDynamicLen(MS);
    }
  }

  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
