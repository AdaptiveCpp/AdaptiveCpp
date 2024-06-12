/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */



#include "hipSYCL/compiler/sscp/StdAtomicRemapperPass.hpp"
#include "hipSYCL/common/debug.hpp"


#include <llvm-15/llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/AtomicOrdering.h>

#include <string>

namespace hipsycl {
namespace compiler {

namespace {

// These enums need to align with the builtin memory scope/order/address-space conventions.
enum class memory_scope : int {
  work_item,
  sub_group,
  work_group,
  device,
  system
};

enum class memory_order : int
{
  relaxed,
  acquire,
  release,
  acq_rel,
  seq_cst
};

enum class address_space : int
{
  global_space,
  local_space,
  constant_space,
  private_space,
  generic_space
};

memory_order llvmOrderingToAcppOrdering(llvm::AtomicOrdering AO) {
  if (AO == llvm::AtomicOrdering::NotAtomic || AO == llvm::AtomicOrdering::Unordered ||
      AO == llvm::AtomicOrdering::Monotonic)
    return memory_order::relaxed;
  // TODO memory_order_consume?
  else if(AO == llvm::AtomicOrdering::Acquire)
    return memory_order::acquire;
  else if(AO == llvm::AtomicOrdering::Release)
    return memory_order::release;
  else if(AO == llvm::AtomicOrdering::AcquireRelease)
    return memory_order::acq_rel;
  else
    return memory_order::seq_cst;
}

llvm::Type* getPointerType(llvm::Type* PointeeT, int AddressSpace) {
#if LLVM_VERSION_MAJOR < 16
    return llvm::PointerType::get(PointeeT, AddressSpace);
#else
    return llvm::PointerType::get(PointeeT->getContext(), AddressSpace);
#endif
}

template<class IntType>
llvm::Value* getIntConstant(llvm::Module& M, IntType Value) {
  return llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), static_cast<uint64_t>(Value));
}

bool isSizeSupportedByBuiltins(llvm::Module& M, llvm::Type* T) {
  auto Size = M.getDataLayout().getTypeSizeInBits(T);
  return Size == 8 || Size == 16 || Size == 32 || Size == 64;
}

bool needsBitcastsForIntAtomics(llvm::Module& M, llvm::Type* T) {
  return isSizeSupportedByBuiltins(M, T) && T->isIntegerTy();
}

llvm::Value *bitcastToIntN(llvm::Module &M, llvm::Value *V, int BitSize,
                           llvm::Instruction *InsertBefore) {
  auto *TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
  return new llvm::BitCastInst(V, TargetType, "", InsertBefore);
}

llvm::Value* ptrcastToIntNPtr(llvm::Module &M, llvm::Value *V, int BitSize,
                           llvm::Instruction *InsertBefore) {
  auto *TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
  return new llvm::BitCastInst(
      V, getPointerType(TargetType, V->getType()->getPointerAddressSpace()), "", InsertBefore);
}

// These functions obtain SSCP builtin declarations in IR

llvm::Function* getAtomicStoreBuiltin(llvm::Module& M, int BitSize) {
  std::string BuiltinName = "__acpp_sscp_atomic_store_i"+std::to_string(BitSize);


  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);

  return static_cast<llvm::Function *>(M.getOrInsertFunction(BuiltinName,
                                                             llvm::Type::getVoidTy(M.getContext()),
                                                             I32Ty, I32Ty, I32Ty, PtrTy, OpTy)
                                           .getCallee());
}


llvm::Function *getAtomicLoadBuiltin(llvm::Module &M, int BitSize) {
  std::string BuiltinName = "__acpp_sscp_atomic_load_i"+std::to_string(BitSize);


  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);
  
  return static_cast<llvm::Function *>(
      M.getOrInsertFunction(BuiltinName, OpTy, I32Ty, I32Ty, I32Ty, PtrTy).getCallee());
}

llvm::Function* getAtomicExchangeBuiltin(llvm::Module& M, int BitSize) {
  std::string BuiltinName = "__acpp_sscp_atomic_exchange_i"+std::to_string(BitSize);

  llvm::Type* I32Ty = llvm::Type::getInt32Ty(M.getContext());
  llvm::Type* OpTy = llvm::Type::getIntNTy(M.getContext(), BitSize);
  llvm::Type* PtrTy = getPointerType(OpTy, 0);
  
  return static_cast<llvm::Function *>(
      M.getOrInsertFunction(BuiltinName, OpTy, I32Ty, I32Ty, I32Ty, PtrTy, OpTy).getCallee());
}

// These functions generate calls to corresponding SSCP builtin declarations

llvm::Value *createAtomicStore(llvm::Module &M, llvm::Value *Value, llvm::Value *Addr,
                               memory_order Order, memory_scope Scope,
                               llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(Value->getType());
  if(!isSizeSupportedByBuiltins(M, Value->getType()))
    return nullptr;

  if(needsBitcastsForIntAtomics(M, Value->getType())) {
    auto* TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
    Value = bitcastToIntN(M, Value, BitSize, InsertBefore);
    Addr = ptrcastToIntNPtr(M, Addr, BitSize, InsertBefore);
  }

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, Order),
    getIntConstant(M, Scope),
    Addr,
    Value
  };

  llvm::Function *Builtin = getAtomicStoreBuiltin(M, BitSize);
  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}


llvm::Value *createAtomicLoad(llvm::Module &M, llvm::Type* DataType, llvm::Value *Addr,
                               memory_order Order, memory_scope Scope,
                               llvm::Instruction *InsertBefore) {
  int BitSize = M.getDataLayout().getTypeSizeInBits(DataType);
  if(!isSizeSupportedByBuiltins(M, DataType))
    return nullptr;

  if(needsBitcastsForIntAtomics(M, DataType)) {
    auto* TargetType = llvm::IntegerType::get(M.getContext(), BitSize);
    Addr = ptrcastToIntNPtr(M, Addr, BitSize, InsertBefore);
  }

  llvm::SmallVector<llvm::Value*> Args {
    getIntConstant(M, address_space::generic_space),
    getIntConstant(M, Order),
    getIntConstant(M, Scope),
    Addr,
  };

  llvm::Function *Builtin = getAtomicLoadBuiltin(M, BitSize);
  return llvm::CallInst::Create(llvm::FunctionCallee(Builtin->getFunctionType(), Builtin),
                                llvm::ArrayRef<llvm::Value *>{Args}, "", InsertBefore);
}


}

llvm::PreservedAnalyses StdAtomicRemapperPass::run(llvm::Module &M,
                                                   llvm::ModuleAnalysisManager &MAM) {

  llvm::SmallVector<llvm::StoreInst*> AtomicStores;
  llvm::SmallVector<llvm::LoadInst*> AtomicLoads;
  

  for(auto& F : M) {
    for(auto& BB : F){
      for(auto& I : BB) {
        if(auto* SI = llvm::dyn_cast<llvm::StoreInst>(&I)) {
          if(SI->isAtomic())
            AtomicStores.push_back(SI);
        } else if(auto* LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
          if(LI->isAtomic())
            AtomicLoads.push_back(LI);
        } else if(auto* XI = llvm::dyn_cast<llvm::AtomicRMWInst>(&I)) {
          
        }
      }
    }
  }

  llvm::SmallVector<llvm::Instruction*> ReplacedInstructions;
  for(auto* AS : AtomicStores) {
    llvm::AtomicOrdering Order = AS->getOrdering();
    if (auto *NewI =
            createAtomicStore(M, AS->getValueOperand(), AS->getPointerOperand(),
                              llvmOrderingToAcppOrdering(Order), memory_scope::device, AS)) {
      AS->replaceNonMetadataUsesWith(NewI);
      ReplacedInstructions.push_back(AS);
    }
  }
  for(auto* AL : AtomicLoads) {
    llvm::AtomicOrdering Order = AL->getOrdering();
    if (auto *NewI =
            createAtomicLoad(M, AL->getType(), AL->getPointerOperand(),
                              llvmOrderingToAcppOrdering(Order), memory_scope::device, AL)) {
      AL->replaceNonMetadataUsesWith(NewI);
      ReplacedInstructions.push_back(AL);
    }
  }

  // Clean up
  for(auto* I : ReplacedInstructions) {
    I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
    I->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}
}
}

