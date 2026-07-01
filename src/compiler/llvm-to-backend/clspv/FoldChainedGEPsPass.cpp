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

// This pass re-folds two-level GEP chains produced by LLVM's instcombine pass
// back into a single typed GEP before the IR is handed to clspv.
//
// Background
// ----------
// LLVM O3's instcombine canonicalises `ptr[n - k]` (where k is a positive
// constant) into a byte-granularity two-GEP chain:
//
//   %inner = getelementptr T,  ptr %base, i64 %n        ; past-end pointer
//   %outer = getelementptr i8, ptr %inner, i64 -(k*sizeof(T))
//
// This form is valid LLVM IR (it avoids the `inbounds` UB that would arise
// from a single `getelementptr inbounds T, ptr %base, i64 (%n-k)` when n==0),
// but clspv --physical-storage-buffers miscompiles it (clspv issue #1292):
// seeing an i8 source-element type, clspv infers 'char' as the pointee type
// and emits four separate byte-wide OpLoads instead of one word-wide OpLoad,
// which produces wrong results or a device crash on Vulkan.
//
// The fold is valid when the constant byte offset of the outer GEP is an exact
// multiple of sizeof(T), which instcombine guarantees for this specific pattern.

#include "hipSYCL/compiler/llvm-to-backend/clspv/FoldChainedGEPsPass.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

namespace {

/// Try to fold a two-level GEP chain:
///   outer = getelementptr i8, ptr inner, i64 C   (C = constant byte offset)
///   inner = getelementptr T,  ptr base,  i64 n
/// into:
///   combined = getelementptr T, ptr base, i64 (n + C/sizeof(T))
///
/// Returns the new GEP on success, nullptr if the pattern does not match or
/// the fold is not safe.
static llvm::GetElementPtrInst *
tryFoldChainedGEP(llvm::GetElementPtrInst *Outer, const llvm::DataLayout &DL) {
  // The outer GEP must be a single-index byte-granularity GEP.
  if (!Outer->getSourceElementType()->isIntegerTy(8))
    return nullptr;
  if (Outer->getNumIndices() != 1)
    return nullptr;

  auto *ByteOffsetCI =
      llvm::dyn_cast<llvm::ConstantInt>(Outer->getOperand(1));
  if (!ByteOffsetCI)
    return nullptr;
  int64_t ByteOffset = ByteOffsetCI->getSExtValue();

  // The pointer operand of the outer GEP must itself be a single-index GEP
  // on a concrete element type (the inner GEP).
  auto *Inner =
      llvm::dyn_cast<llvm::GetElementPtrInst>(Outer->getPointerOperand());
  if (!Inner)
    return nullptr;
  if (Inner->getNumIndices() != 1)
    return nullptr;

  llvm::Type *ElemTy = Inner->getSourceElementType();
  if (ElemTy->isVoidTy() || !ElemTy->isSized())
    return nullptr;

  llvm::TypeSize ElemTSz = DL.getTypeStoreSize(ElemTy);
  if (ElemTSz.isScalable() || ElemTSz.getFixedValue() == 0)
    return nullptr;

  int64_t ElemBytes = static_cast<int64_t>(ElemTSz.getFixedValue());

  // The fold is lossless only when ByteOffset is an exact multiple of the
  // element size. (instcombine guarantees this for its specific transform.)
  if (ByteOffset % ElemBytes != 0)
    return nullptr;

  int64_t IndexDelta = ByteOffset / ElemBytes; // may be negative

  // Build the combined GEP immediately before the outer GEP so that
  // dominance is maintained for the base pointer and original index.
  llvm::IRBuilder<> Builder(Outer);
  llvm::Value *OrigIndex = Inner->getOperand(1);

  llvm::Value *NewIndex;
  if (IndexDelta == 0) {
    NewIndex = OrigIndex;
  } else {
    llvm::Type *IdxTy = OrigIndex->getType();
    NewIndex =
        Builder.CreateAdd(OrigIndex,
                          llvm::ConstantInt::get(IdxTy,
                                                 static_cast<uint64_t>(IndexDelta),
                                                 /*isSigned=*/true),
                          "gep.fold.idx");
  }

  llvm::Value *NewGEPVal = Builder.CreateGEP(ElemTy, Inner->getPointerOperand(),
                                             {NewIndex}, Outer->getName());
  auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(NewGEPVal);

  // Only mark the combined GEP as inbounds when both source GEPs were inbounds.
  // A negative byte offset means the outer GEP is not inbounds (it accesses
  // memory before the inner pointer), so this is a conservative approach.
  if (Inner->isInBounds() && Outer->isInBounds())
    NewGEP->setIsInBounds(true);

  return NewGEP;
}

} // namespace

llvm::PreservedAnalyses
FoldChainedGEPsPass::run(llvm::Function &F,
                         llvm::FunctionAnalysisManager &) {
  const llvm::DataLayout &DL = F.getParent()->getDataLayout();

  // Collect GEPs to fold first to avoid iterator invalidation.
  llvm::SmallVector<llvm::GetElementPtrInst *, 16> OuterGEPs;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
        OuterGEPs.push_back(GEP);
    }
  }

  llvm::SmallVector<llvm::GetElementPtrInst *, 16> ToErase;
  bool Changed = false;

  for (auto *Outer : OuterGEPs) {
    // Skip if already erased (e.g. was an inner GEP of a previous fold).
    if (!Outer->getParent())
      continue;

    // Remember the inner GEP before we replace Outer (so we can check if
    // it becomes dead after the replacement).
    auto *InnerGEP =
        llvm::dyn_cast<llvm::GetElementPtrInst>(Outer->getPointerOperand());

    auto *NewGEP = tryFoldChainedGEP(Outer, DL);
    if (!NewGEP)
      continue;

    Outer->replaceAllUsesWith(NewGEP);
    ToErase.push_back(Outer);

    // If the inner GEP is now dead (Outer was its only user), schedule removal.
    if (InnerGEP && InnerGEP->use_empty())
      ToErase.push_back(InnerGEP);

    Changed = true;
  }

  // Erase in reverse order so that a use is removed before its def when both
  // Outer (use of Inner) and Inner are in ToErase.
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It) {
    auto *I = *It;
    if (I->getParent() && I->use_empty())
      I->eraseFromParent();
  }

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
