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


#include "hipSYCL/compiler/reflection/IntrospectStructPass.hpp"
#include "hipSYCL/compiler/utils/AggregateTypeUtils.hpp"
#include "hipSYCL/common/debug.hpp"
#include <climits>
#include <cstdint>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/ADT/SmallVector.h>



namespace hipsycl {
namespace compiler {

namespace {
constexpr const char* builtin_name = "__hipsycl_introspect_flattened_struct";

struct TypeInformation {
  int FlattenedNumMembers;
  llvm::SmallVector<int, 8> MemberOffsets;
  llvm::SmallVector<int, 8> MemberSizes;
  llvm::SmallVector<int, 8> MemberKind;
};

enum class TypeKind { Other = 0, Pointer = 1, IntegerType = 2, FloatType = 3 };

TypeKind getTypeKind(llvm::Type* T) {
  if(T->isPointerTy())
    return TypeKind::Pointer;
  else if(T->isFloatingPointTy())
    return TypeKind::FloatType;
  else if(T->isIntegerTy())
    return TypeKind::IntegerType;
  return TypeKind::Other;
}

TypeInformation getTypeInformation(llvm::Type* T, llvm::Module& M) {
  TypeInformation TI;
  TI.FlattenedNumMembers = 0;

  auto OnContainedType = [&](llvm::Type *ContainedT, llvm::SmallVector<int, 16> Indices) {
    llvm::SmallVector<llvm::Value *> GEPIndices;
    for (int Idx : Indices)
      GEPIndices.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), Idx));
    llvm::ArrayRef<llvm::Value *> GEPIndicesRef{GEPIndices};
    std::size_t Offset = M.getDataLayout().getIndexedOffsetInType(T, GEPIndicesRef);
    std::size_t ByteSize = M.getDataLayout().getTypeSizeInBits(ContainedT) / CHAR_BIT;
  
    TypeKind Kind = getTypeKind(ContainedT);

    TI.FlattenedNumMembers++;
    TI.MemberOffsets.push_back(Offset);
    TI.MemberSizes.push_back(ByteSize);
    TI.MemberKind.push_back(static_cast<int>(Kind));
  };

  // The {0} as initial index allows us to directly use the indexing for GetElementPtr-like
  // offset calculations
  utils::ForEachNonAggregateContainedType(T, OnContainedType, {0});
  return TI;
}

struct TypeInformationGlobalVars {
  llvm::GlobalVariable* FlattenedNumMembers;
  llvm::GlobalVariable* MemberOffsets;
  llvm::GlobalVariable* MemberSizes;
  llvm::GlobalVariable* MemberKind;
};

template<unsigned N>
llvm::Constant* createConstantIntArray(const llvm::SmallVector<int, N>& Data, llvm::Module& M) {
  llvm::SmallVector<llvm::Constant*, N> Fields;
  for(int i = 0; i < Data.size(); ++i) {
    Fields.push_back(llvm::Constant::getIntegerValue(
        llvm::IntegerType::getInt32Ty(M.getContext()),
        llvm::APInt{32, static_cast<uint64_t>(Data[i])}));
  }
  return llvm::ConstantArray::get(
      llvm::ArrayType::get(llvm::IntegerType::getInt32Ty(M.getContext()), Fields.size()),
      Fields);
}

TypeInformationGlobalVars storeTypeInformationAsGlobals(llvm::Function *BuiltinInstantiation,
                                                        llvm::Module &M,
                                                        const TypeInformation &TI) {
  TypeInformationGlobalVars TIGV;

  std::string GVPrefix = "__hipsycl_typeinfo_" + BuiltinInstantiation->getName().str() + "_";

  llvm::GlobalVariable *FlattenedNumMembersGV =
      new llvm::GlobalVariable(M, llvm::IntegerType::getInt32Ty(M.getContext()), true,
                               llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
                               llvm::Constant::getIntegerValue(
                                   llvm::IntegerType::getInt32Ty(M.getContext()),
                                   llvm::APInt{32, static_cast<uint64_t>(TI.FlattenedNumMembers)}),
                               GVPrefix + "flattened_num_members", nullptr,
                               llvm::GlobalValue::ThreadLocalMode::NotThreadLocal);

  auto *IntArrayType = llvm::ArrayType::get(llvm::IntegerType::getInt32Ty(M.getContext()),
                                            static_cast<uint64_t>(TI.FlattenedNumMembers));
  llvm::GlobalVariable *MemberOffsetsGV =
      new llvm::GlobalVariable(M, IntArrayType, true,
                               llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
                               createConstantIntArray(TI.MemberOffsets, M),
                               GVPrefix + "member_offsets", nullptr,
                               llvm::GlobalValue::ThreadLocalMode::NotThreadLocal);

  llvm::GlobalVariable *MemberSizesGV =
      new llvm::GlobalVariable(M, IntArrayType, true,
                               llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
                               createConstantIntArray(TI.MemberSizes, M),
                               GVPrefix + "member_sizes", nullptr,
                               llvm::GlobalValue::ThreadLocalMode::NotThreadLocal);

  llvm::GlobalVariable *MemberKindGV =
      new llvm::GlobalVariable(M, IntArrayType, true,
                               llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage,
                               createConstantIntArray(TI.MemberKind, M),
                               GVPrefix + "member_kind", nullptr,
                               llvm::GlobalValue::ThreadLocalMode::NotThreadLocal);

  TIGV.FlattenedNumMembers = FlattenedNumMembersGV;
  TIGV.MemberOffsets = MemberOffsetsGV;
  TIGV.MemberSizes = MemberSizesGV;
  TIGV.MemberKind = MemberKindGV;

  return TIGV;
}

llvm::AllocaInst *recurseOperandUntilAlloca(llvm::Instruction *I) {
  if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(I))
    return AI;

  if (I->getNumOperands() == 0)
    return nullptr;

  if (auto *AI = llvm::dyn_cast<llvm::AllocaInst>(I->getOperand(0))) {
    return AI;
  } else if (auto *BI = llvm::dyn_cast<llvm::BitCastInst>(I->getOperand(0))) {
    return recurseOperandUntilAlloca(BI);
  } else if (auto *GEPI = llvm::dyn_cast<llvm::GetElementPtrInst>(I->getOperand(0))) {
    return recurseOperandUntilAlloca(GEPI);
  } else {
    return nullptr;
  }
}

} // anonymous namespace

llvm::PreservedAnalyses IntrospectStructPass::run(llvm::Module& M, llvm::ModuleAnalysisManager& AM) {
  llvm::SmallVector<llvm::Function*, 8> BuiltinInstantiations;
  for(auto& F: M) {
    if(F.getName().contains(builtin_name)) {
      BuiltinInstantiations.push_back(&F);
    }
  }

  llvm::SmallDenseMap<llvm::Function *, TypeInformation> StructTypeInfoForBuiltin;
  llvm::SmallVector<llvm::CallInst *, 16> Calls;
  for (auto *F : BuiltinInstantiations) {
    for (auto *U : F->users()) {
      if (llvm::CallInst *CI = llvm::dyn_cast<llvm::CallInst>(U)) {
        Calls.push_back(CI);

        if (auto *AI = recurseOperandUntilAlloca(CI)) {
          llvm::Type *T = AI->getAllocatedType();
          StructTypeInfoForBuiltin[F] = getTypeInformation(T, M);
        } else {
          HIPSYCL_DEBUG_WARNING
              << "IntrospectStructPass: " << F->getName().str()
              << " instantiation could not be connected to type information, ignoring.\n";
        }
      }
    }
  }

  if (StructTypeInfoForBuiltin.size() > 0)
    for(auto* CI : Calls) {
      llvm::Function* F = CI->getCalledFunction();
      if(F) {
        auto It = StructTypeInfoForBuiltin.find(F);

        auto GVs = storeTypeInformationAsGlobals(F, M, It->getSecond());

        if(It != StructTypeInfoForBuiltin.end()) {
          if(F->getFunctionType()->getNumParams() != 5) {
            HIPSYCL_DEBUG_WARNING << "IntrospectStructPass: Call to " << F->getName().str()
            << " has the wrong number of parameters, ignoring call\n";
          } else {
            auto createStoreOp = [&](int ArgIndex, llvm::GlobalVariable* GV) {
              if(!CI->getArgOperand(ArgIndex)->getType()->isPointerTy()) {
                HIPSYCL_DEBUG_WARNING
                << "IntrospectStructPass: Call to " << F->getName().str()
                << " is invalid; argument is not a pointer type. Ingoring call.\n";
                return;
              }
              [[maybe_unused]] llvm::StoreInst *S =
                new llvm::StoreInst(GV, CI->getArgOperand(ArgIndex), CI);
            };

            createStoreOp(1, GVs.FlattenedNumMembers);
            createStoreOp(2, GVs.MemberOffsets);
            createStoreOp(3, GVs.MemberSizes);
            createStoreOp(4, GVs.MemberKind);
          }
        }
      }
    }

  // Lastly, remove calls and builtin declarations from IR
  for(auto* CI : Calls)
    CI->eraseFromParent();
  for(auto* F : BuiltinInstantiations) {
    F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
    F->eraseFromParent();  
  }

  return llvm::PreservedAnalyses::none();
}

}
}
