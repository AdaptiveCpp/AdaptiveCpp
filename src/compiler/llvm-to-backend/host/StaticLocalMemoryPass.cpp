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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/Support/Casting.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/StaticLocalMemoryPass.hpp"
#include "hipSYCL/compiler/utils/ConstantExpressions.hpp"

namespace hipsycl::compiler {

namespace {
bool checkCapacity(std::size_t Position, std::size_t Capacity) {
  if (Position > Capacity) {
    HIPSYCL_DEBUG_ERROR << "[LLVMToHost] Processing of static local memory exceeded maximum static "
                           "local memory size of "
                        << Capacity
                        << "; Please reduce the size of statically requested local memory.\n";
    return false;
  }
  return true;
}

static const char *InternalLocalMemBuiltinName = "__acpp_sscp_host_get_internal_local_memory";

llvm::Value *prependInternalLocalMemAccessCall(llvm::Module &M, llvm::Type *TargetPtrType,
                                               llvm::Function *F, std::size_t Offset,
                                               unsigned LocalMemAS) {
#if LLVM_VERSION_MAJOR >= 16
  auto *VoidPtrType = llvm::PointerType::get(M.getContext(), 0);
#else
  auto *VoidPtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), 0);
#endif

  auto Builtin = M.getOrInsertFunction(InternalLocalMemBuiltinName, VoidPtrType);
  assert(Builtin);

  auto *InsertionPt = &(*F->getEntryBlock().getFirstInsertionPt());
  llvm::Instruction *BuiltinCall =
      llvm::CallInst::Create(llvm::FunctionCallee{Builtin}, "", InsertionPt);
  // user will expect AS3, so cast back to AS3
#if LLVM_VERSION_MAJOR < 17
  auto *AS3PtrType = llvm::PointerType::getWithSamePointeeType(VoidPtrType, LocalMemAS);
#else
  auto *AS3PtrType = llvm::PointerType::get(M.getContext(), LocalMemAS);
#endif
  auto *ASCastInst = new llvm::AddrSpaceCastInst{BuiltinCall, AS3PtrType, "", InsertionPt};

  // GEP to index at the offset into the array
  auto OffsetInt = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M.getContext()), Offset);
  llvm::SmallVector<llvm::Value *> GEPIndices{OffsetInt};
  auto *GEP = llvm::GetElementPtrInst::CreateInBounds(
      llvm::Type::getInt8Ty(M.getContext()), ASCastInst, llvm::ArrayRef<llvm::Value *>{GEPIndices},
      "", InsertionPt);

  auto *BI = llvm::BitCastInst::Create(llvm::Instruction::BitCast, GEP, TargetPtrType, "", InsertionPt);

  return BI;
}


void replaceGVWithInternalLocalMem(llvm::GlobalVariable* GV, llvm::Module& M, std::size_t Offset) {
  const unsigned LocalMemAS = 3;

  // If the GV is not directly used by instructions but instead used by
  // constant expressions (which might then be used by instructions),
  // we need to unfold the CEs first.
  llvm::SmallVector<llvm::ConstantExpr*> CEUsers;
  for(auto* U : GV->users()) {
    if(auto* CE = llvm::dyn_cast<llvm::ConstantExpr>(U))
      CEUsers.push_back(CE);
  }
  for(auto* CE : CEUsers)
    utils::unfoldConstantExpression(CE);

  llvm::SmallDenseMap<llvm::Function *, llvm::Value *> FunctionToLocalMemMap;

  for (auto *U : GV->users()) {
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(U)) {
      llvm::Function *F = I->getParent()->getParent();
      if (FunctionToLocalMemMap.find(F) == FunctionToLocalMemMap.end()) {
        if (!F->isDeclaration()) {
          FunctionToLocalMemMap[F] =
              prependInternalLocalMemAccessCall(M, GV->getType(), F, Offset, LocalMemAS);
        }
      }
    }
  }

  for (auto Entry : FunctionToLocalMemMap) {
    llvm::Function *TargetF = Entry.first;
    llvm::Value *ReplacementI = Entry.second;

    GV->replaceUsesWithIf(ReplacementI, [&](llvm::Use &U) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U.getUser())) {
        if (I->getParent() && I->getParent()->getParent()) {
          auto *OwningF = I->getParent()->getParent();
          if (OwningF == TargetF) {
            return true;
          }
        }
      }
      return false;
    });
  }
}

} // namespace

llvm::PreservedAnalyses HostStaticLocalMemoryPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  // These parameters need to be aligned with the kernel call and allocation
  // the local memory block in omp_queue.cpp!
  std::size_t Offset = 1024 * sizeof(uint64_t);
  std::size_t MaxSize = 64 * sizeof(uint64_t) * 1024;

  for(llvm::GlobalVariable& GV : M.globals()) {
    if(GV.getAddressSpace() == 3 && GV.getLinkage() != llvm::GlobalValue::LinkageTypes::ExternalLinkage) {

      auto Alignment = GV.getAlign().valueOrOne().value();
      if(Offset % Alignment != 0) {
        Offset = ((Offset + Alignment - 1) / Alignment) * Alignment;
      }

      std::size_t Size = M.getDataLayout().getTypeSizeInBits(GV.getValueType()) / CHAR_BIT;

      if(!checkCapacity(Offset + Size, MaxSize)) {
        return llvm::PreservedAnalyses::none();
      }

      replaceGVWithInternalLocalMem(&GV, M, Offset);

      Offset += Size;
    }
  }

  return llvm::PreservedAnalyses::none();
}

}