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

#include "hipSYCL/compiler/sscp/AggregateArgumentExpansionPass.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Constants.h>

namespace hipsycl {
namespace compiler {

namespace {

template <class F>
void ForEachNonAggregateContainedType(llvm::Type *T, F &&Handler,
                                      llvm::SmallVector<int, 16> CurrentIndices) {
  if(!T)
    return;
  
  if(T->isArrayTy()) {
    llvm::Type* ArrayElementT = T->getArrayElementType();
    for(int i = 0; i < T->getArrayNumElements(); ++i) {
      auto NextIndices = CurrentIndices;
      NextIndices.push_back(i);
      ForEachNonAggregateContainedType(ArrayElementT, Handler, NextIndices);
    }
  } else if(T->isAggregateType()) {
    for(int i = 0; i < T->getNumContainedTypes(); ++i) {
      auto NextIndices = CurrentIndices;
      NextIndices.push_back(i);
      llvm::Type* SubType = T->getContainedType(i);

      ForEachNonAggregateContainedType(SubType, Handler, NextIndices);   
    }
  } else {
    Handler(T, CurrentIndices);
  }
}

llvm::Type* getValueType(llvm::Function& F, int ArgNo) {
  llvm::Type* ArgT = F.getFunctionType()->getParamType(ArgNo);
  if(ArgT->isPointerTy() && F.hasParamAttribute(ArgNo, llvm::Attribute::ByVal)) {
    llvm::Type *ValT = F.getParamAttribute(ArgNo, llvm::Attribute::ByVal).getValueAsType();
    return ValT;
  } else {
    return ArgT;
  }
}

bool needsExpansion(llvm::Function& F, int ArgNo) {
  llvm::Type* ArgT = F.getFunctionType()->getParamType(ArgNo);
  if(ArgT->isPointerTy() && F.hasParamAttribute(ArgNo, llvm::Attribute::ByVal))
    return true;
  if(ArgT->isAggregateType())
    return true;
  return false;
}

struct ExpandedArgumentInfo {
  // Original value type
  llvm::Type* OriginalByValType = nullptr;
  
  // Indices for getelementptr to all expanded arguments of the original
  // arg
  llvm::SmallVector<llvm::SmallVector<int, 16>> GEPIndices;
  llvm::SmallVector<llvm::Type*> ExpandedTypes;
  // original argument index
  int OriginalIndex = 0;
  // Number of arguments the original arg was expanded into
  int NumExpandedArguments = 1;
  bool IsExpanded = false;
};

void ExpandAggregateArguments(llvm::Module& M, llvm::Function& F) {
  
  // stores one entry per *original* function argument. Maps
  // original argument index to expansion info.
  llvm::SmallDenseMap<int, ExpandedArgumentInfo> ExpansionInfo;

  for(int i = 0; i < F.getFunctionType()->getNumParams(); ++i) {
    ExpandedArgumentInfo Info;
    Info.OriginalIndex = i;

    if (needsExpansion(F, i)) {
      llvm::SmallVector<int, 16> InitialContainedTypeIndices;
      InitialContainedTypeIndices.push_back(0);

      Info.IsExpanded = true;

      auto OnContainedType = [&](llvm::Type *T, llvm::SmallVector<int, 16> Indices) {
        Info.ExpandedTypes.push_back(T);
        Info.GEPIndices.push_back(Indices);
      };

      auto* ValueT = getValueType(F, i);
      ForEachNonAggregateContainedType(ValueT, OnContainedType, InitialContainedTypeIndices);
      Info.OriginalByValType = ValueT;
      Info.NumExpandedArguments = Info.GEPIndices.size();
    } else {
      auto* ArgT = F.getFunctionType()->getParamType(i);
      Info.OriginalByValType = ArgT;
    }

    ExpansionInfo[i] = Info;
  }

  llvm::SmallVector<llvm::Type*, 16> NewArgumentTypes;
  for(int i = 0; i < F.getFunctionType()->getNumParams(); ++i) {
    auto& EI = ExpansionInfo[i];
    if(EI.IsExpanded) {
      for(auto* T : EI.ExpandedTypes)
        NewArgumentTypes.push_back(T);
    } else {
      NewArgumentTypes.push_back(EI.OriginalByValType);
    }
  }
  
  std::string FunctionName = F.getName().str();
  F.setName(FunctionName + "_PreArgumentExpansion");
  auto OldLinkage = F.getLinkage();
  F.setLinkage(llvm::GlobalValue::InternalLinkage);

  llvm::FunctionType *FType = llvm::FunctionType::get(F.getReturnType(), NewArgumentTypes, false);
  if (auto *NewF = llvm::dyn_cast<llvm::Function>(
          M.getOrInsertFunction(FunctionName, FType).getCallee())) {
    for(auto& Attr : F.getAttributes().getFnAttrs()) {
      NewF->addFnAttr(Attr);
    }
    NewF->setLinkage(OldLinkage);

    llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", NewF);
    
    llvm::SmallVector<llvm::Value*, 8> CallArgs;

    int CurrentNewIndex = 0;
    for(int i = 0; i < F.getFunctionType()->getNumParams(); ++i) {
      auto& EI = ExpansionInfo[i];
      if(EI.IsExpanded) {
        // We need to reconstruct the aggregate that was decomposed.
        // First, allocate some memory
        auto* Alloca = new llvm::AllocaInst(EI.OriginalByValType, 0, "", BB);
        // Iterate over all elements that the aggregate was expanded into:
        for(int j = 0; j < EI.NumExpandedArguments; ++j) {
          // Create getelementptr instruction to offset into the allocated struct
          llvm::SmallVector<llvm::Value*> GEPIndices;
          for(int Idx : EI.GEPIndices[j]) {
            GEPIndices.push_back(
                llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), Idx));
          }

          auto *GEPInst = llvm::GetElementPtrInst::CreateInBounds(
              EI.OriginalByValType, Alloca,
              llvm::ArrayRef<llvm::Value *>{GEPIndices}, "", BB);
          // Store expanded argument into allocated space
          assert(CurrentNewIndex + j < NewF->getFunctionType()->getNumParams());
          
          auto *StoredVal = NewF->getArg(CurrentNewIndex + j);
          auto *StoreInst = new llvm::StoreInst(StoredVal, GEPInst, BB);
        }
        CallArgs.push_back(Alloca);
      } else {
        CallArgs.push_back(NewF->getArg(CurrentNewIndex));
      }
      CurrentNewIndex += EI.NumExpandedArguments;
    }
    
    auto *Call = llvm::CallInst::Create(llvm::FunctionCallee(&F), CallArgs,
                                            "", BB);
    for(int i = 0; i < CallArgs.size(); ++i) {
      if(F.hasParamAttribute(i, llvm::Attribute::ByVal))
        Call->addParamAttr(i, F.getParamAttribute(i, llvm::Attribute::ByVal));
    }
    llvm::ReturnInst::Create(M.getContext(), BB);

    if(!F.hasFnAttribute(llvm::Attribute::AlwaysInline))
      F.addFnAttr(llvm::Attribute::AlwaysInline);
  }
}

}

AggregateArgumentExpansionPass::AggregateArgumentExpansionPass(
    const std::vector<std::string> &FunctioNames)
    : AffectedFunctionNames{FunctioNames} {}

llvm::PreservedAnalyses AggregateArgumentExpansionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  for(const auto& FN : AffectedFunctionNames) {
    if(auto* F = M.getFunction(FN)) {
      ExpandAggregateArguments(M, *F);
    }
  }
  return llvm::PreservedAnalyses::none();
}

}
}
