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

#include "hipSYCL/compiler/llvm-to-backend/clspv/LegalizeIntWidthsPass.hpp"
#include "hipSYCL/common/debug.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueHandle.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Transforms/Utils/Local.h>

namespace hipsycl {
namespace compiler {

namespace {

bool isLegalWidth(unsigned BitWidth) {
  return BitWidth == 1 || BitWidth == 8 || BitWidth == 16 || BitWidth == 32 ||
         BitWidth == 64;
}

bool isIllegalIntTy(const llvm::Type *T) {
  return T->isIntegerTy() && !isLegalWidth(T->getIntegerBitWidth());
}

/// Smallest legal width that can hold \p BitWidth bits, 0 if there is none.
unsigned getPromotedWidth(unsigned BitWidth) {
  for (unsigned W : {8u, 16u, 32u, 64u}) {
    if (BitWidth <= W)
      return W;
  }
  return 0;
}

llvm::IntegerType *getPromotedTy(llvm::Type *T) {
  unsigned W = getPromotedWidth(T->getIntegerBitWidth());
  assert(W != 0 && "No legal type to promote to");
  return llvm::IntegerType::get(T->getContext(), W);
}

struct MemoryChunk {
  uint64_t OffsetInBytes;
  unsigned Bits;
};

/// Decomposes an object of \p BitWidth bits into chunks of legal width,
/// starting with the widest chunk. The offset of every chunk is a multiple of
/// its own size, so that each access can be expressed as a typed GEP.
llvm::SmallVector<MemoryChunk, 4> computeChunks(unsigned BitWidth) {
  llvm::SmallVector<MemoryChunk, 4> Chunks;
  uint64_t RemainingBytes = (BitWidth + 7) / 8;
  uint64_t Offset = 0;
  while (RemainingBytes > 0) {
    uint64_t ChunkBytes = 8;
    while (ChunkBytes > RemainingBytes) {
      ChunkBytes /= 2;
    }
    Chunks.push_back({Offset, static_cast<unsigned>(ChunkBytes * 8)});
    Offset += ChunkBytes;
    RemainingBytes -= ChunkBytes;
  }
  return Chunks;
}

llvm::Value *getChunkPtr(llvm::IRBuilder<> &B, llvm::Value *BasePtr,
                         const MemoryChunk &C, llvm::Type *ChunkTy) {
  if (C.OffsetInBytes == 0) {
    return BasePtr;
  }
  const uint64_t ChunkBytes = C.Bits / 8;
  assert(C.OffsetInBytes % ChunkBytes == 0 && "Chunk offset not aligned");
  return B.CreateConstInBoundsGEP1_64(ChunkTy, BasePtr,
                                      C.OffsetInBytes / ChunkBytes);
}

llvm::Align getChunkAlign(llvm::Align BaseAlign, const MemoryChunk &C) {
  if (C.OffsetInBytes == 0) {
    return BaseAlign;
  }
  return llvm::commonAlignment(BaseAlign, C.OffsetInBytes);
}

bool isSupportedOpcode(const llvm::Instruction *I) {
  switch (I->getOpcode()) {
  case llvm::Instruction::Load:
    return llvm::cast<llvm::LoadInst>(I)->isSimple();
  case llvm::Instruction::Store:
    return llvm::cast<llvm::StoreInst>(I)->isSimple();
  case llvm::Instruction::PHI:
  case llvm::Instruction::Select:
  case llvm::Instruction::ICmp:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::BitCast:
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
    return true;
  default:
    return false;
  }
}

/// Scalar illegal integers either fit a legal type or use the i128 limb map.
bool isPromotableIllegalTy(const llvm::Type *T) {
  return isIllegalIntTy(T) && (T->isIntegerTy(128) ||
                               getPromotedWidth(T->getIntegerBitWidth()) != 0);
}

bool usesI128(const llvm::Instruction &I) {
  if (I.getType()->isIntegerTy(128)) {
    return true;
  }
  for (const llvm::Use &U : I.operands()) {
    if (U->getType()->isIntegerTy(128)) {
      return true;
    }
  }
  return false;
}

bool isSupportedI128(const llvm::Instruction &I, const llvm::DataLayout &DL) {
  switch (I.getOpcode()) {
  case llvm::Instruction::Load:
    return llvm::cast<llvm::LoadInst>(I).isSimple();
  case llvm::Instruction::Store:
    return llvm::cast<llvm::StoreInst>(I).isSimple();
  case llvm::Instruction::PHI:
  case llvm::Instruction::Select:
  case llvm::Instruction::ICmp:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
    return true;
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt: {
    llvm::Type *Other = I.getType()->isIntegerTy(128)
                            ? I.getOperand(0)->getType()
                            : I.getType();
    return Other->isIntegerTy() && Other->getIntegerBitWidth() <= 64;
  }
  case llvm::Instruction::BitCast: {
    llvm::Type *Other = I.getType()->isIntegerTy(128)
                            ? I.getOperand(0)->getType()
                            : I.getType();
    auto *VT = llvm::dyn_cast<llvm::FixedVectorType>(Other);
    if (!VT || DL.getTypeSizeInBits(VT) != 128) {
      return false;
    }
    llvm::Type *Element = VT->getElementType();
    return (Element->isIntegerTy() &&
            isLegalWidth(Element->getIntegerBitWidth())) ||
           Element->isHalfTy() || Element->isFloatTy() || Element->isDoubleTy();
  }
  default:
    return false;
  }
}

bool hasIllegalOperand(const llvm::Instruction *I) {
  for (const llvm::Use &U : I->operands()) {
    if (isIllegalIntTy(U.get()->getType())) {
      return true;
    }
  }
  return false;
}

class IntWidthLegalizer {
public:
  IntWidthLegalizer(llvm::Function &F)
      : F{F}, DL{F.getParent()->getDataLayout()} {}

  /// Returns true if the function was modified.
  bool run();

private:
  /// Replaces stores of values too wide to be promoted by a sequence of chunk
  /// sized stores.
  bool splitWideStores();

  /// Materializes the bits [\p LoBit, \p LoBit + \p Bits) of \p V as a value
  /// of type i\p Bits, without ever creating a value of illegal width.
  ///
  /// Returns null if the bits cannot be extracted, in which case no
  /// instructions have been added that are not trivially dead.
  llvm::Value *extractBits(llvm::IRBuilder<> &B, llvm::Value *V, unsigned LoBit,
                           unsigned Bits);

  bool collectWork();
  bool rewrite();

  /// Value of promoted type whose low `N` bits match the low `N` bits of \p V.
  /// Bits above `N` are undefined.
  llvm::Value *getPromoted(llvm::Value *V);

  llvm::Value *maskToWidth(llvm::IRBuilder<> &B, llvm::Value *PromotedV,
                           unsigned Width);
  llvm::Value *signExtendWithin(llvm::IRBuilder<> &B, llvm::Value *PromotedV,
                                unsigned Width);
  /// Adjusts a promoted value to \p TargetTy, preserving its low bits.
  llvm::Value *adjustWidth(llvm::IRBuilder<> &B, llvm::Value *V,
                           llvm::Type *TargetTy);

  llvm::Value *emitSplitLoad(llvm::IRBuilder<> &B, llvm::LoadInst *LI,
                             llvm::Type *CombineTy);
  void emitSplitStore(llvm::IRBuilder<> &B, llvm::StoreInst *SI,
                      llvm::Value *PromotedVal);

  llvm::Value *rewriteInstruction(llvm::Instruction *I);
  struct Limbs {
    llvm::Value *Lo;
    llvm::Value *Hi;
  };
  Limbs getLimbs(llvm::Value *V);
  llvm::Value *rewriteI128(llvm::Instruction *I);
  Limbs shiftLimbs(llvm::IRBuilder<> &B, Limbs V, Limbs Amount,
                   unsigned Opcode);

  /// Makes sure that phi nodes created by the pass are well formed, even if
  /// legalization was aborted half way through.
  void completeCreatedPhis();

  llvm::Function &F;
  const llvm::DataLayout &DL;

  llvm::SmallVector<llvm::Instruction *, 32> Work;
  llvm::SmallVector<llvm::AllocaInst *, 8> IllegalAllocas;
  llvm::DenseMap<llvm::Value *, llvm::Value *> Promoted;
  llvm::DenseMap<llvm::PHINode *, llvm::PHINode *> PhiMap;
  llvm::DenseMap<llvm::Value *, Limbs> Wide;
  llvm::DenseMap<llvm::PHINode *, Limbs> WidePhis;
};

llvm::Value *IntWidthLegalizer::extractBits(llvm::IRBuilder<> &B,
                                            llvm::Value *V, unsigned LoBit,
                                            unsigned Bits) {
  auto *ResTy = llvm::IntegerType::get(F.getContext(), Bits);
  const unsigned Width = V->getType()->getIntegerBitWidth();
  assert(LoBit + Bits <= Width && "Requested bits outside of value");

  if (llvm::isa<llvm::UndefValue>(V)) {
    return llvm::UndefValue::get(ResTy);
  }

  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    return llvm::ConstantInt::get(ResTy,
                                  CI->getValue().lshr(LoBit).trunc(Bits));
  }

  // A plain load of a wide value: load the requested chunk directly.
  if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(V)) {
    if (!LI->isSimple() || LoBit % 8 != 0 || Bits % 8 != 0 ||
        !isLegalWidth(Bits)) {
      return nullptr;
    }
    const MemoryChunk C{LoBit / 8, Bits};
    if (C.OffsetInBytes % (Bits / 8) != 0) {
      return nullptr;
    }
    llvm::IRBuilder<> LB{LI};
    llvm::Value *Ptr = getChunkPtr(LB, LI->getPointerOperand(), C, ResTy);
    return LB.CreateAlignedLoad(ResTy, Ptr, getChunkAlign(LI->getAlign(), C));
  }

  if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
    switch (I->getOpcode()) {
    case llvm::Instruction::ZExt: {
      // Bits at or above the source width are known to be zero.
      llvm::Value *Src = I->getOperand(0);
      const unsigned SrcWidth = Src->getType()->getIntegerBitWidth();
      if (LoBit >= SrcWidth) {
        return llvm::ConstantInt::get(ResTy, 0);
      }
      const unsigned SubBits = std::min(Bits, SrcWidth - LoBit);
      if (!isLegalWidth(SubBits)) {
        return nullptr;
      }
      llvm::Value *Sub = extractBits(B, Src, LoBit, SubBits);
      if (!Sub) {
        return nullptr;
      }
      return SubBits == Bits ? Sub : B.CreateZExt(Sub, ResTy);
    }
    case llvm::Instruction::Trunc: {
      llvm::Value *Src = I->getOperand(0);
      // Truncation only drops high bits, so the requested bits are unchanged.
      return extractBits(B, Src, LoBit, Bits);
    }
    case llvm::Instruction::Shl: {
      auto *Amount = llvm::dyn_cast<llvm::ConstantInt>(I->getOperand(1));
      if (!Amount) {
        return nullptr;
      }
      const uint64_t Sh = Amount->getZExtValue();
      // Bits below the shift amount are zero.
      if (LoBit + Bits <= Sh) {
        return llvm::ConstantInt::get(ResTy, 0);
      }
      if (LoBit >= Sh) {
        return extractBits(B, I->getOperand(0), LoBit - Sh, Bits);
      }
      llvm::Value *Sub = extractBits(B, I->getOperand(0), 0, Bits);
      if (!Sub) {
        return nullptr;
      }
      return B.CreateShl(Sub, llvm::ConstantInt::get(ResTy, Sh - LoBit));
    }
    case llvm::Instruction::LShr: {
      auto *Amount = llvm::dyn_cast<llvm::ConstantInt>(I->getOperand(1));
      if (!Amount) {
        return nullptr;
      }
      const uint64_t Sh = Amount->getZExtValue();
      if (LoBit + Sh >= Width) {
        return llvm::ConstantInt::get(ResTy, 0);
      }
      if (LoBit + Bits + Sh > Width) {
        return nullptr;
      }
      return extractBits(B, I->getOperand(0), LoBit + Sh, Bits);
    }
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor: {
      // Bitwise operations act on each bit independently.
      llvm::Value *L = extractBits(B, I->getOperand(0), LoBit, Bits);
      if (!L) {
        return nullptr;
      }
      llvm::Value *R = extractBits(B, I->getOperand(1), LoBit, Bits);
      if (!R) {
        return nullptr;
      }
      return B.CreateBinOp(
          static_cast<llvm::Instruction::BinaryOps>(I->getOpcode()), L, R);
    }
    case llvm::Instruction::Select: {
      llvm::Value *T = extractBits(B, I->getOperand(1), LoBit, Bits);
      if (!T) {
        return nullptr;
      }
      llvm::Value *Fa = extractBits(B, I->getOperand(2), LoBit, Bits);
      if (!Fa) {
        return nullptr;
      }
      return B.CreateSelect(I->getOperand(0), T, Fa);
    }
    default:
      break;
    }
  }

  // Opaque value: it can only be taken apart if it is of legal width itself.
  if (!isLegalWidth(Width)) {
    return nullptr;
  }
  llvm::Value *Shifted =
      LoBit == 0 ? V
                 : B.CreateLShr(V, llvm::ConstantInt::get(V->getType(), LoBit));
  return Width == Bits ? Shifted : B.CreateTrunc(Shifted, ResTy);
}

// Phase 0: Rewrite stores of values that are too wide to be promoted.
//
// Values wider than 64 bits have no legal type they could be promoted to, so
// the only way to legalize them is to never materialize them: the stored value
// is decomposed into chunks of legal width that are stored individually. This
// works whenever every chunk can be computed without constructing a wide value
// itself, which `extractBits()` checks.
//
// Typical sources of such values are `sycl::vec<float, 3/4>` return values of
// relational builtins, which SROA packs into a single `i128` built from a tree
// of `zext`/`shl`/`or`, and plain copies of wide objects.
bool IntWidthLegalizer::splitWideStores() {
  llvm::SmallVector<llvm::StoreInst *, 8> WideStores;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto *SI = llvm::dyn_cast<llvm::StoreInst>(&I);
      if (!SI || !SI->isSimple()) {
        continue;
      }
      llvm::Type *ValTy = SI->getValueOperand()->getType();
      if (isIllegalIntTy(ValTy) &&
          getPromotedWidth(ValTy->getIntegerBitWidth()) == 0) {
        WideStores.push_back(SI);
      }
    }
  }

  bool Changed = false;
  llvm::SmallVector<llvm::WeakTrackingVH, 16> MaybeDead;
  for (llvm::StoreInst *SI : WideStores) {
    llvm::Value *Val = SI->getValueOperand();
    const unsigned BitWidth = Val->getType()->getIntegerBitWidth();
    auto Chunks = computeChunks(BitWidth);

    // Build the chunk values first: if any of them cannot be extracted the
    // store is left alone, and the instructions created so far are dead and
    // cleaned up below.
    llvm::IRBuilder<> B{SI};
    llvm::SmallVector<llvm::Value *, 4> ChunkValues;
    bool Extracted = true;
    for (const MemoryChunk &C : Chunks) {
      llvm::Value *ChunkVal = extractBits(B, Val, C.OffsetInBytes * 8, C.Bits);
      if (!ChunkVal) {
        Extracted = false;
        break;
      }
      ChunkValues.push_back(ChunkVal);
    }

    for (llvm::Value *V : ChunkValues)
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(V))
        MaybeDead.push_back(I);

    if (!Extracted)
      continue;

    for (size_t i = 0; i < Chunks.size(); ++i) {
      const MemoryChunk &C = Chunks[i];
      auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
      llvm::Value *DstPtr = getChunkPtr(B, SI->getPointerOperand(), C, ChunkTy);
      B.CreateAlignedStore(ChunkValues[i], DstPtr,
                           getChunkAlign(SI->getAlign(), C));
    }

    if (auto *ValI = llvm::dyn_cast<llvm::Instruction>(Val))
      MaybeDead.push_back(ValI);
    SI->eraseFromParent();
    Changed = true;
  }

  // Remove the now dead wide value computations, as they would otherwise still
  // contain illegal types and cause the legalization below to bail out.
  llvm::RecursivelyDeleteTriviallyDeadInstructionsPermissive(MaybeDead);

  return Changed;
}

// Phase 2: Scan the function in reverse post-order and record every
// instruction that needs rewriting, without modifying the IR.
bool IntWidthLegalizer::collectWork() {
  llvm::ReversePostOrderTraversal<llvm::Function *> RPO{&F};
  for (llvm::BasicBlock *BB : RPO) {
    for (llvm::Instruction &I : *BB) {
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (isIllegalIntTy(AI->getAllocatedType())) {
          IllegalAllocas.push_back(AI);
        }
      }

      if (!isIllegalIntTy(I.getType()) && !hasIllegalOperand(&I)) {
        continue;
      }

      if (!isSupportedOpcode(&I) || (usesI128(I) && !isSupportedI128(I, DL)) ||
          (I.getOpcode() == llvm::Instruction::BitCast && !usesI128(I))) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Unsupported instruction using an "
               "integer type of non-standard bit width in function "
            << F.getName().str() << "; clspv may reject this kernel\n";
        return false;
      }

      auto reject = [&](const char *Reason) {
        HIPSYCL_DEBUG_WARNING << "LegalizeIntWidthsPass: " << Reason
                              << " in function " << F.getName().str() << " ("
                              << I.getOpcodeName()
                              << "); clspv may reject this kernel\n";
      };

      // Computations must fit into a legal type or use the i128 limb mapping.
      if (isIllegalIntTy(I.getType()) && !isPromotableIllegalTy(I.getType())) {
        reject("Integer type too wide to be promoted to a legal width");
        return false;
      }
      for (llvm::Use &U : I.operands()) {
        llvm::Type *OpTy = U.get()->getType();
        if (isIllegalIntTy(OpTy) && !isPromotableIllegalTy(OpTy)) {
          reject("Operand of an integer type too wide to be promoted to a "
                 "legal width");
          return false;
        }
        if (isIllegalIntTy(OpTy) && !llvm::isa<llvm::Instruction>(U.get()) &&
            !llvm::isa<llvm::ConstantInt>(U.get()) &&
            !llvm::isa<llvm::UndefValue>(U.get())) {
          reject("Operand of non-standard integer width with no legalizable "
                 "counterpart");
          return false;
        }
      }

      Work.push_back(&I);
    }
  }
  return true;
}

// Returns the legal-width counterpart of an illegal-width value.
//
// For instructions this is the entry recorded by phase 3; thanks to the RPO
// order it is already present for every operand, with the exception of phi
// back edges which are resolved after the main loop. Constants and undef are
// materialised on demand. A null return means the value cannot be legalized
// and aborts the rewrite.
llvm::Value *IntWidthLegalizer::getPromoted(llvm::Value *V) {
  assert(isIllegalIntTy(V->getType()) && "Value is already of legal type");

  auto It = Promoted.find(V);
  if (It != Promoted.end()) {
    return It->second;
  }

  llvm::IntegerType *PromTy = getPromotedTy(V->getType());
  if (auto *CI = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    return llvm::ConstantInt::get(PromTy,
                                  CI->getValue().zext(PromTy->getBitWidth()));
  }
  if (llvm::isa<llvm::PoisonValue>(V)) {
    return llvm::PoisonValue::get(PromTy);
  }
  if (llvm::isa<llvm::UndefValue>(V)) {
    return llvm::UndefValue::get(PromTy);
  }

  return nullptr;
}

llvm::Value *IntWidthLegalizer::maskToWidth(llvm::IRBuilder<> &B,
                                            llvm::Value *PromotedV,
                                            unsigned Width) {
  const unsigned PromWidth = PromotedV->getType()->getIntegerBitWidth();
  if (PromWidth == Width) {
    return PromotedV;
  }
  return B.CreateAnd(
      PromotedV,
      llvm::ConstantInt::get(PromotedV->getType(),
                             llvm::APInt::getLowBitsSet(PromWidth, Width)));
}

llvm::Value *IntWidthLegalizer::signExtendWithin(llvm::IRBuilder<> &B,
                                                 llvm::Value *PromotedV,
                                                 unsigned Width) {
  const unsigned PromWidth = PromotedV->getType()->getIntegerBitWidth();
  if (PromWidth == Width) {
    return PromotedV;
  }
  llvm::Value *ShiftAmount =
      llvm::ConstantInt::get(PromotedV->getType(), PromWidth - Width);
  return B.CreateAShr(B.CreateShl(PromotedV, ShiftAmount), ShiftAmount);
}

llvm::Value *IntWidthLegalizer::adjustWidth(llvm::IRBuilder<> &B,
                                            llvm::Value *V,
                                            llvm::Type *TargetTy) {
  const unsigned FromWidth = V->getType()->getIntegerBitWidth();
  const unsigned ToWidth = TargetTy->getIntegerBitWidth();
  if (FromWidth == ToWidth) {
    return V;
  }
  if (FromWidth > ToWidth) {
    return B.CreateTrunc(V, TargetTy);
  }
  return B.CreateZExt(V, TargetTy);
}

IntWidthLegalizer::Limbs IntWidthLegalizer::getLimbs(llvm::Value *V) {
  auto It = Wide.find(V);
  if (It != Wide.end()) {
    return It->second;
  }
  auto *Ty = llvm::Type::getInt64Ty(F.getContext());
  if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    return {llvm::ConstantInt::get(Ty, C->getValue().trunc(64)),
            llvm::ConstantInt::get(Ty, C->getValue().lshr(64).trunc(64))};
  }
  if (llvm::isa<llvm::PoisonValue>(V)) {
    return {llvm::PoisonValue::get(Ty), llvm::PoisonValue::get(Ty)};
  }
  if (llvm::isa<llvm::UndefValue>(V)) {
    return {llvm::UndefValue::get(Ty), llvm::UndefValue::get(Ty)};
  }
  llvm_unreachable("Missing i128 counterpart");
}

IntWidthLegalizer::Limbs IntWidthLegalizer::shiftLimbs(llvm::IRBuilder<> &B,
                                                       Limbs V, Limbs Amount,
                                                       unsigned Opcode) {
  auto *Ty = B.getInt64Ty();
  auto C = [&](uint64_t N) { return llvm::ConstantInt::get(Ty, N); };
  // Every emitted i64 shift is in [0, 63], including the unused select arms.
  // In particular, the cross-limb contribution at shift zero must be zero,
  // not a shift by 64 (which would make the whole result poison).
  llvm::Value *N = B.CreateAnd(Amount.Lo, C(63));
  llvm::Value *Reverse = B.CreateAnd(B.CreateSub(C(64), N), C(63));
  llvm::Value *Zero = B.CreateICmpEQ(N, C(0));
  llvm::Value *Small = B.CreateICmpULT(Amount.Lo, C(64));
  llvm::Value *Valid = B.CreateAnd(B.CreateICmpEQ(Amount.Hi, C(0)),
                                   B.CreateICmpULT(Amount.Lo, C(128)));
  llvm::Value *Lo;
  llvm::Value *Hi;
  if (Opcode == llvm::Instruction::Shl) {
    llvm::Value *Cross =
        B.CreateSelect(Zero, C(0), B.CreateLShr(V.Lo, Reverse));
    Lo = B.CreateSelect(Small, B.CreateShl(V.Lo, N), C(0));
    Hi = B.CreateSelect(Small, B.CreateOr(B.CreateShl(V.Hi, N), Cross),
                        B.CreateShl(V.Lo, N));
  } else {
    const bool Signed = Opcode == llvm::Instruction::AShr;
    llvm::Value *HighShift =
        Signed ? B.CreateAShr(V.Hi, N) : B.CreateLShr(V.Hi, N);
    llvm::Value *Cross = B.CreateSelect(Zero, C(0), B.CreateShl(V.Hi, Reverse));
    Lo = B.CreateSelect(Small, B.CreateOr(B.CreateLShr(V.Lo, N), Cross),
                        HighShift);
    Hi = B.CreateSelect(Small, HighShift,
                        Signed ? B.CreateAShr(V.Hi, C(63)) : C(0));
  }
  // An i128 source is a single value: poison in either limb must also reach a
  // limb that the shift otherwise fills with zeros. Bitwise masking does not
  // introduce poison for undef, unlike a self-comparison used as a guard.
  llvm::Value *SourceZero = B.CreateAnd(B.CreateOr(V.Lo, V.Hi), C(0));
  Lo = B.CreateOr(Lo, SourceZero);
  Hi = B.CreateOr(Hi, SourceZero);
  auto *Poison = llvm::PoisonValue::get(Ty);
  return {B.CreateSelect(Valid, Lo, Poison), B.CreateSelect(Valid, Hi, Poison)};
}

llvm::Value *IntWidthLegalizer::rewriteI128(llvm::Instruction *I) {
  llvm::IRBuilder<> B{I};
  auto *Ty = B.getInt64Ty();
  auto C = [&](uint64_t N) { return llvm::ConstantInt::get(Ty, N); };
  Limbs Result{nullptr, nullptr};
  switch (I->getOpcode()) {
  case llvm::Instruction::Load:
  case llvm::Instruction::Store: {
    auto *LI = llvm::dyn_cast<llvm::LoadInst>(I);
    auto *SI = llvm::dyn_cast<llvm::StoreInst>(I);
    llvm::Value *Ptr = LI ? LI->getPointerOperand() : SI->getPointerOperand();
    llvm::Align Alignment = LI ? LI->getAlign() : SI->getAlign();
    if (SI) {
      Result = getLimbs(SI->getValueOperand());
    }
    for (unsigned Index = 0; Index != 2; ++Index) {
      MemoryChunk Chunk{Index * 8u, 64};
      llvm::Value *ChunkPtr = getChunkPtr(B, Ptr, Chunk, Ty);
      llvm::Value *&Limb =
          (Index == 0) == DL.isLittleEndian() ? Result.Lo : Result.Hi;
      llvm::Instruction *Access;
      if (LI) {
        auto *Load =
            B.CreateAlignedLoad(Ty, ChunkPtr, getChunkAlign(Alignment, Chunk));
        Limb = Load;
        Access = Load;
      } else {
        Access = B.CreateAlignedStore(Limb, ChunkPtr,
                                      getChunkAlign(Alignment, Chunk));
      }
      // Value/size-specific metadata (range, tbaa.struct, etc.) cannot be
      // copied unchanged onto narrower accesses.
      Access->copyMetadata(
          *I, {llvm::LLVMContext::MD_tbaa, llvm::LLVMContext::MD_alias_scope,
               llvm::LLVMContext::MD_noalias, llvm::LLVMContext::MD_nontemporal,
               llvm::LLVMContext::MD_access_group,
               llvm::LLVMContext::MD_invariant_load,
               llvm::LLVMContext::MD_invariant_group});
    }
    if (SI) {
      return nullptr;
    }
    break;
  }
  case llvm::Instruction::PHI: {
    auto *Phi = llvm::cast<llvm::PHINode>(I);
    Result = {llvm::PHINode::Create(Ty, Phi->getNumIncomingValues(),
                                    Phi->getName() + ".lo", Phi),
              llvm::PHINode::Create(Ty, Phi->getNumIncomingValues(),
                                    Phi->getName() + ".hi", Phi)};
    WidePhis[Phi] = Result;
    break;
  }
  case llvm::Instruction::Select: {
    Limbs T = getLimbs(I->getOperand(1));
    Limbs E = getLimbs(I->getOperand(2));
    Result = {B.CreateSelect(I->getOperand(0), T.Lo, E.Lo),
              B.CreateSelect(I->getOperand(0), T.Hi, E.Hi)};
    break;
  }
  case llvm::Instruction::ICmp: {
    Limbs L = getLimbs(I->getOperand(0));
    Limbs R = getLimbs(I->getOperand(1));
    auto Pred = llvm::cast<llvm::ICmpInst>(I)->getPredicate();
    if (Pred == llvm::CmpInst::ICMP_EQ)
      return B.CreateAnd(B.CreateICmpEQ(L.Lo, R.Lo),
                         B.CreateICmpEQ(L.Hi, R.Hi));
    if (Pred == llvm::CmpInst::ICMP_NE)
      return B.CreateOr(B.CreateICmpNE(L.Lo, R.Lo), B.CreateICmpNE(L.Hi, R.Hi));
    auto Strict = llvm::CmpInst::getStrictPredicate(Pred);
    auto LowPred = llvm::ICmpInst::getUnsignedPredicate(Pred);
    return B.CreateOr(B.CreateICmp(Strict, L.Hi, R.Hi),
                      B.CreateAnd(B.CreateICmpEQ(L.Hi, R.Hi),
                                  B.CreateICmp(LowPred, L.Lo, R.Lo)));
  }
  case llvm::Instruction::Trunc: {
    llvm::Type *Dst = I->getType();
    if (isIllegalIntTy(Dst)) {
      Dst = getPromotedTy(Dst);
    }
    return adjustWidth(B, getLimbs(I->getOperand(0)).Lo, Dst);
  }
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt: {
    llvm::Value *Src = I->getOperand(0);
    unsigned Width = Src->getType()->getIntegerBitWidth();
    if (isIllegalIntTy(Src->getType())) {
      Src = getPromoted(Src);
    }
    bool Signed = I->getOpcode() == llvm::Instruction::SExt;
    Src = Signed ? signExtendWithin(B, Src, Width) : maskToWidth(B, Src, Width);
    Result.Lo = Signed ? B.CreateSExtOrTrunc(Src, Ty) : adjustWidth(B, Src, Ty);
    Result.Hi = Signed ? B.CreateAShr(Result.Lo, C(63)) : C(0);
    break;
  }
  case llvm::Instruction::BitCast: {
    auto *VT = llvm::FixedVectorType::get(Ty, 2);
    if (I->getType()->isIntegerTy(128)) {
      llvm::Value *V = B.CreateBitCast(I->getOperand(0), VT);
      Result = {
          B.CreateExtractElement(V, B.getInt32(DL.isLittleEndian() ? 0 : 1)),
          B.CreateExtractElement(V, B.getInt32(DL.isLittleEndian() ? 1 : 0))};
    } else {
      Limbs V = getLimbs(I->getOperand(0));
      llvm::Value *Vector = llvm::PoisonValue::get(VT);
      Vector = B.CreateInsertElement(Vector, V.Lo,
                                     B.getInt32(DL.isLittleEndian() ? 0 : 1));
      Vector = B.CreateInsertElement(Vector, V.Hi,
                                     B.getInt32(DL.isLittleEndian() ? 1 : 0));
      return B.CreateBitCast(Vector, I->getType());
    }
    break;
  }
  default: {
    Limbs L = getLimbs(I->getOperand(0));
    Limbs R = getLimbs(I->getOperand(1));
    auto *Op = llvm::cast<llvm::BinaryOperator>(I);
    unsigned Opcode = I->getOpcode();
    if (Opcode == llvm::Instruction::Shl || Opcode == llvm::Instruction::LShr ||
        Opcode == llvm::Instruction::AShr) {
      Result = shiftLimbs(B, L, R, Opcode);
      llvm::Value *Valid = B.getTrue();
      auto equal = [&](Limbs A, Limbs E) {
        return B.CreateAnd(B.CreateICmpEQ(A.Lo, E.Lo),
                           B.CreateICmpEQ(A.Hi, E.Hi));
      };
      if (Op->isExact()) {
        Valid = equal(shiftLimbs(B, Result, R, llvm::Instruction::Shl), L);
      }
      if (Op->hasNoUnsignedWrap()) {
        Valid = B.CreateAnd(
            Valid, equal(shiftLimbs(B, Result, R, llvm::Instruction::LShr), L));
      }
      if (Op->hasNoSignedWrap()) {
        Valid = B.CreateAnd(
            Valid, equal(shiftLimbs(B, Result, R, llvm::Instruction::AShr), L));
      }
      if (Op->isExact() || Op->hasNoUnsignedWrap() || Op->hasNoSignedWrap()) {
        auto *Poison = llvm::PoisonValue::get(Ty);
        Result = {B.CreateSelect(Valid, Result.Lo, Poison),
                  B.CreateSelect(Valid, Result.Hi, Poison)};
      }
    } else {
      Result = {B.CreateBinOp(Op->getOpcode(), L.Lo, R.Lo),
                B.CreateBinOp(Op->getOpcode(), L.Hi, R.Hi)};
    }
    break;
  }
  }
  Wide[I] = Result;
  return Result.Lo;
}

llvm::Value *IntWidthLegalizer::emitSplitLoad(llvm::IRBuilder<> &B,
                                              llvm::LoadInst *LI,
                                              llvm::Type *CombineTy) {
  const unsigned BitWidth = LI->getType()->getIntegerBitWidth();
  auto Chunks = computeChunks(BitWidth);

  llvm::Value *Result = nullptr;
  for (const MemoryChunk &C : Chunks) {
    auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
    llvm::Value *Ptr = getChunkPtr(B, LI->getPointerOperand(), C, ChunkTy);
    llvm::Value *ChunkVal =
        B.CreateAlignedLoad(ChunkTy, Ptr, getChunkAlign(LI->getAlign(), C));
    llvm::Value *Extended = adjustWidth(B, ChunkVal, CombineTy);
    unsigned Shift = DL.isLittleEndian() ? C.OffsetInBytes * 8
                                         : ((BitWidth + 7) / 8) * 8 -
                                               C.OffsetInBytes * 8 - C.Bits;
    if (Shift != 0) {
      Extended =
          B.CreateShl(Extended, llvm::ConstantInt::get(CombineTy, Shift));
    }
    Result = Result ? B.CreateOr(Result, Extended) : Extended;
  }
  return Result;
}

void IntWidthLegalizer::emitSplitStore(llvm::IRBuilder<> &B,
                                       llvm::StoreInst *SI,
                                       llvm::Value *PromotedVal) {
  const unsigned BitWidth =
      SI->getValueOperand()->getType()->getIntegerBitWidth();
  auto Chunks = computeChunks(BitWidth);

  // Bits above the original width are undefined in the promoted value; mask
  // them off so that no garbage is written to memory.
  llvm::Value *Val = maskToWidth(B, PromotedVal, BitWidth);
  for (const MemoryChunk &C : Chunks) {
    auto *ChunkTy = llvm::IntegerType::get(F.getContext(), C.Bits);
    llvm::Value *Ptr = getChunkPtr(B, SI->getPointerOperand(), C, ChunkTy);
    llvm::Value *ChunkVal = Val;
    unsigned Shift = DL.isLittleEndian() ? C.OffsetInBytes * 8
                                         : ((BitWidth + 7) / 8) * 8 -
                                               C.OffsetInBytes * 8 - C.Bits;
    if (Shift != 0) {
      ChunkVal =
          B.CreateLShr(ChunkVal, llvm::ConstantInt::get(Val->getType(), Shift));
    }
    ChunkVal = adjustWidth(B, ChunkVal, ChunkTy);
    B.CreateAlignedStore(ChunkVal, Ptr, getChunkAlign(SI->getAlign(), C));
  }
}

llvm::Value *IntWidthLegalizer::rewriteInstruction(llvm::Instruction *I) {
  if (usesI128(*I)) {
    return rewriteI128(I);
  }
  llvm::IRBuilder<> B{I};

  auto promotedOperand = [&](unsigned Idx) -> llvm::Value * {
    llvm::Value *Op = I->getOperand(Idx);
    return isIllegalIntTy(Op->getType()) ? getPromoted(Op) : Op;
  };

  switch (I->getOpcode()) {
  case llvm::Instruction::Load: {
    auto *LI = llvm::cast<llvm::LoadInst>(I);
    return emitSplitLoad(B, LI, getPromotedTy(LI->getType()));
  }
  case llvm::Instruction::Store: {
    auto *SI = llvm::cast<llvm::StoreInst>(I);
    llvm::Value *Val = promotedOperand(0);
    if (!Val) {
      return nullptr;
    }
    emitSplitStore(B, SI, Val);
    return nullptr;
  }
  case llvm::Instruction::PHI: {
    // Incoming values are patched once all instructions have been rewritten.
    auto *Phi = llvm::cast<llvm::PHINode>(I);
    auto *NewPhi = llvm::PHINode::Create(getPromotedTy(Phi->getType()),
                                         Phi->getNumIncomingValues(),
                                         Phi->getName() + ".legalized", Phi);
    PhiMap[Phi] = NewPhi;
    return NewPhi;
  }
  case llvm::Instruction::Select: {
    llvm::Value *T = promotedOperand(1);
    llvm::Value *Fa = promotedOperand(2);
    if (!T || !Fa) {
      return nullptr;
    }
    return B.CreateSelect(I->getOperand(0), T, Fa);
  }
  case llvm::Instruction::ICmp: {
    auto *Cmp = llvm::cast<llvm::ICmpInst>(I);
    const unsigned Width = Cmp->getOperand(0)->getType()->getIntegerBitWidth();
    llvm::Value *L = promotedOperand(0);
    llvm::Value *R = promotedOperand(1);
    if (!L || !R) {
      return nullptr;
    }
    if (Cmp->isSigned()) {
      L = signExtendWithin(B, L, Width);
      R = signExtendWithin(B, R, Width);
    } else {
      L = maskToWidth(B, L, Width);
      R = maskToWidth(B, R, Width);
    }
    return B.CreateICmp(Cmp->getPredicate(), L, R);
  }
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt: {
    llvm::Value *Src = promotedOperand(0);
    if (!Src) {
      return nullptr;
    }
    llvm::Type *SrcTy = I->getOperand(0)->getType();
    llvm::Type *DstTy = I->getType();
    llvm::Type *NewDstTy = isIllegalIntTy(DstTy) ? getPromotedTy(DstTy) : DstTy;

    if (isIllegalIntTy(SrcTy)) {
      const unsigned SrcWidth = SrcTy->getIntegerBitWidth();
      if (I->getOpcode() == llvm::Instruction::ZExt) {
        Src = maskToWidth(B, Src, SrcWidth);
      } else if (I->getOpcode() == llvm::Instruction::SExt) {
        Src = signExtendWithin(B, Src, SrcWidth);
      }
    }

    const unsigned SrcPromWidth = Src->getType()->getIntegerBitWidth();
    const unsigned DstPromWidth = NewDstTy->getIntegerBitWidth();
    if (SrcPromWidth == DstPromWidth) {
      return Src;
    }
    if (SrcPromWidth > DstPromWidth) {
      return B.CreateTrunc(Src, NewDstTy);
    }
    // Widening a legal source keeps the original signedness, widening an
    // already sign/zero extended illegal source can use either.
    if (I->getOpcode() == llvm::Instruction::SExt) {
      return B.CreateSExt(Src, NewDstTy);
    }
    return B.CreateZExt(Src, NewDstTy);
  }
  default:
    break;
  }

  auto *BinOp = llvm::dyn_cast<llvm::BinaryOperator>(I);
  if (!BinOp) {
    return nullptr;
  }

  const unsigned Width = BinOp->getType()->getIntegerBitWidth();
  llvm::Value *L = promotedOperand(0);
  llvm::Value *R = promotedOperand(1);
  if (!L || !R) {
    return nullptr;
  }

  switch (BinOp->getOpcode()) {
  case llvm::Instruction::Add:
  case llvm::Instruction::Sub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Shl:
    // The low 'Width' bits of the result only depend on the low 'Width' bits
    // of the operands.
    break;
  case llvm::Instruction::LShr:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::URem:
    L = maskToWidth(B, L, Width);
    R = maskToWidth(B, R, Width);
    break;
  case llvm::Instruction::AShr:
    L = signExtendWithin(B, L, Width);
    R = maskToWidth(B, R, Width);
    break;
  case llvm::Instruction::SDiv:
  case llvm::Instruction::SRem:
    L = signExtendWithin(B, L, Width);
    R = signExtendWithin(B, R, Width);
    break;
  default:
    return nullptr;
  }

  return B.CreateBinOp(BinOp->getOpcode(), L, R);
}

void IntWidthLegalizer::completeCreatedPhis() {
  for (auto &Entry : PhiMap) {
    llvm::PHINode *Old = Entry.first;
    llvm::PHINode *New = Entry.second;
    while (New->getNumIncomingValues() < Old->getNumIncomingValues()) {
      const unsigned Idx = New->getNumIncomingValues();
      New->addIncoming(llvm::PoisonValue::get(New->getType()),
                       Old->getIncomingBlock(Idx));
    }
  }
  for (auto &Entry : WidePhis) {
    llvm::PHINode *Old = Entry.first;
    for (llvm::Value *Limb : {Entry.second.Lo, Entry.second.Hi}) {
      auto *New = llvm::cast<llvm::PHINode>(Limb);
      while (New->getNumIncomingValues() < Old->getNumIncomingValues()) {
        const unsigned Idx = New->getNumIncomingValues();
        New->addIncoming(llvm::PoisonValue::get(New->getType()),
                         Old->getIncomingBlock(Idx));
      }
    }
  }
}

// Phase 3 and 4: rewrite all collected instructions in RPO order, patch the
// phi nodes created along the way, then tear down the originals.
bool IntWidthLegalizer::rewrite() {
  // Phase 3. Operands are guaranteed to be legalized already (RPO), so a
  // single ordered pass is sufficient; no fixpoint iteration is required.
  for (llvm::Instruction *I : Work) {
    llvm::Value *New = rewriteInstruction(I);
    if (!New) {
      if (!I->getType()->isVoidTy()) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Could not legalize instruction in "
               "function "
            << F.getName().str() << "; clspv may reject this kernel\n";
        completeCreatedPhis();
        return false;
      }
      continue;
    }
    if (I->getType()->isIntegerTy(128)) {
      continue;
    }
    if (isIllegalIntTy(I->getType())) {
      Promoted[I] = New;
    } else {
      I->replaceAllUsesWith(New);
    }
  }

  // Loop back edges are the only place where RPO does not guarantee that an
  // operand has been rewritten before its user, so the new phi nodes were
  // created with no incoming values. Now that every value has a legalized
  // counterpart, fill them in.
  for (auto &Entry : PhiMap) {
    llvm::PHINode *Old = Entry.first;
    llvm::PHINode *New = Entry.second;
    for (unsigned i = 0; i < Old->getNumIncomingValues(); ++i) {
      llvm::Value *In = Old->getIncomingValue(i);
      llvm::Value *NewIn = isIllegalIntTy(In->getType()) ? getPromoted(In) : In;
      if (!NewIn) {
        completeCreatedPhis();
        return false;
      }
      New->addIncoming(NewIn, Old->getIncomingBlock(i));
    }
  }
  for (auto &Entry : WidePhis) {
    llvm::PHINode *Old = Entry.first;
    for (unsigned i = 0; i < Old->getNumIncomingValues(); ++i) {
      Limbs In = getLimbs(Old->getIncomingValue(i));
      llvm::cast<llvm::PHINode>(Entry.second.Lo)
          ->addIncoming(In.Lo, Old->getIncomingBlock(i));
      llvm::cast<llvm::PHINode>(Entry.second.Hi)
          ->addIncoming(In.Hi, Old->getIncomingBlock(i));
    }
  }

  // Phase 4. The originals may still reference each other, and phi nodes in
  // loops form reference cycles, so drop all references first to break those
  // cycles before erasing. Both loops run in reverse work-list order, i.e.
  // users before definitions.
  for (llvm::Instruction *I : llvm::reverse(Work)) {
    I->dropAllReferences();
  }
  for (llvm::Instruction *I : llvm::reverse(Work)) {
    I->eraseFromParent();
  }

  return true;
}

bool IntWidthLegalizer::run() {
  // Detect i128 before the legacy store splitter can mutate the function.
  bool HasI128 = false;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      HasI128 |= usesI128(I);
    }
  }
  bool Changed = false;

  // Phase 1. Illegal types in the signature would require rewriting the
  // function itself along with all of its call sites, which is out of scope:
  // by the time this pass runs the backend has aggressively inlined the
  // module, so such signatures are not expected in practice.
  auto rejectSignature = [&]() {
    HIPSYCL_DEBUG_WARNING
        << "LegalizeIntWidthsPass: Signature of function " << F.getName().str()
        << " uses an integer type of non-standard bit width; clspv may reject "
           "this kernel\n";
  };
  for (const llvm::Argument &A : F.args()) {
    if (isIllegalIntTy(A.getType())) {
      rejectSignature();
      return Changed;
    }
  }
  if (isIllegalIntTy(F.getReturnType())) {
    rejectSignature();
    return Changed;
  }

  // i128 lowering is transactional: do not let the legacy tree extractor
  // mutate stores before rejecting an unsupported use elsewhere.
  if (!HasI128) {
    Changed = splitWideStores();
  } else {
    llvm::ReversePostOrderTraversal<llvm::Function *> RPO{&F};
    llvm::SmallPtrSet<llvm::BasicBlock *, 16> Reachable;
    for (llvm::BasicBlock *BB : RPO) {
      Reachable.insert(BB);
    }
    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB)
        if ((usesI128(I) && !isSupportedI128(I, DL)) ||
            (!Reachable.count(&BB) &&
             (isIllegalIntTy(I.getType()) || hasIllegalOperand(&I)))) {
          HIPSYCL_DEBUG_WARNING
              << "LegalizeIntWidthsPass: Unsupported i128 use in function "
              << F.getName().str() << "; clspv may reject this kernel\n";
          return false;
        }
    }
  }
  // Phase 2.
  if (!collectWork()) {
    return Changed;
  }

  // Phase 3 and 4.
  if (!Work.empty()) {
    if (!rewrite()) {
      HIPSYCL_DEBUG_WARNING
          << "LegalizeIntWidthsPass: Failed to legalize integer widths in "
             "function "
          << F.getName().str() << "\n";
    }
    Changed = true;
  }

  // Phase 5. Allocas of illegal types are replaced by equally sized byte
  // arrays; all accesses to them have been split into legal accesses above.
  // Since pointers are opaque, users of the alloca do not need to be rewritten
  // beyond that.
  for (llvm::AllocaInst *AI : IllegalAllocas) {
    const uint64_t SizeInBytes = DL.getTypeStoreSize(AI->getAllocatedType());
    llvm::IRBuilder<> B{AI};
    auto *NewTy = llvm::ArrayType::get(llvm::Type::getInt8Ty(F.getContext()),
                                       SizeInBytes);
    auto *NewAlloca = B.CreateAlloca(NewTy, AI->getAddressSpace(),
                                     AI->getArraySize(), AI->getName());
    NewAlloca->setAlignment(AI->getAlign());
    AI->replaceAllUsesWith(NewAlloca);
    AI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

/// Phase 6: Reports integer types of non-standard width that survived
/// legalization, as those would result in an obscure clspv error message later
/// on. Unlike the RPO walk of phase 2 this visits all basic blocks, so it also
/// covers unreachable code the rewrite does not reach.
void warnOnRemainingIllegalTypes(llvm::Function &F) {
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      bool Illegal = isIllegalIntTy(I.getType());
      for (llvm::Use &U : I.operands()) {
        Illegal |= isIllegalIntTy(U.get()->getType());
      }
      if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        Illegal |= isIllegalIntTy(AI->getAllocatedType());
      }

      if (Illegal) {
        HIPSYCL_DEBUG_WARNING
            << "LegalizeIntWidthsPass: Function " << F.getName().str()
            << " still contains integer types of non-standard bit width, "
               "which clspv cannot lower to SPIR-V\n";
        return;
      }
    }
  }
}

} // namespace

llvm::PreservedAnalyses
LegalizeIntWidthsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  if (F.isDeclaration()) {
    return llvm::PreservedAnalyses::all();
  }

  IntWidthLegalizer Legalizer{F};
  const bool Changed = Legalizer.run();

  warnOnRemainingIllegalTypes(F);

  return Changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
