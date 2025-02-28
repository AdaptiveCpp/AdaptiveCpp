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

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/llvm-to-backend/host/StaticLocalMemoryPass.hpp"

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

void replaceGVWithInternalLocalMem(llvm::GlobalVariable* GV, llvm::Module& M, std::size_t Offset) {
  const unsigned LocalMemAS = 3;

#if LLVM_VERSION_MAJOR >= 16
  auto *VoidPtrType = llvm::PointerType::get(M.getContext(), 0);
#else
  auto *VoidPtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), 0);
#endif
  
  auto Builtin = M.getOrInsertFunction(InternalLocalMemBuiltinName, VoidPtrType);
  assert(Builtin);
  
  llvm::SmallPtrSet<llvm::Use*, 16> GVUses;
  for(auto& U : GV->uses()) {
    GVUses.insert(&U);
  }

  for(auto* U : GVUses) {
    if(auto* I = llvm::dyn_cast<llvm::Instruction>(U->getUser())) {
      auto* Call = llvm::CallInst::Create(llvm::FunctionCallee(Builtin),
                                 "", I);
      // user will expect AS3, so cast back to AS3
#if LLVM_VERSION_MAJOR < 17
      auto *AS3PtrType = llvm::PointerType::getWithSamePointeeType(VoidPtrType, LocalMemAS);
#else
      auto *AS3PtrType = llvm::PointerType::get(M.getContext(), LocalMemAS);
#endif
      auto *ASCastInst = new llvm::AddrSpaceCastInst{Call, AS3PtrType, "", I};


      // GEP to index at the offset into the array
      auto OffsetInt = llvm::ConstantInt::get(llvm::Type::getInt64Ty(M.getContext()), Offset);
      llvm::SmallVector<llvm::Value *> GEPIndices{OffsetInt};
      auto* GEP = llvm::GetElementPtrInst::CreateInBounds(llvm::Type::getInt8Ty(M.getContext()), ASCastInst,
                                              llvm::ArrayRef<llvm::Value *>{GEPIndices}, "", I);

      auto *TargetPtrType = U->get()->getType();
      auto *BI = llvm::BitCastInst::Create(llvm::Instruction::BitCast, GEP, TargetPtrType, "", I);

      I->replaceAllUsesWith(BI);
      I->dropAllReferences();
      I->eraseFromParent();
    }
  }
}

} // namespace

llvm::PreservedAnalyses HostStaticLocalMemoryPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  // These parameters need to be aligned with the kernel call and allocation
  // the local memory block in omp_queue.cpp!
  std::size_t Offset = 1024 * sizeof(uint64_t);
  std::size_t MaxSize = 32768 * sizeof(uint64_t);

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