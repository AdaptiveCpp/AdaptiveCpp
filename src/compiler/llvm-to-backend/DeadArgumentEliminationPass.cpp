/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/llvm-to-backend/DeadArgumentEliminationPass.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Constants.h>

namespace hipsycl {
namespace compiler {

namespace {

void removeUnusedFunctionParameters(llvm::Function *F, llvm::Module &M,
                                    llvm::SmallVector<int> *RetainedParameterIndices = nullptr) {

  std::string FunctionName = F->getName().str();
  F->setName(FunctionName+".old");
  llvm::SmallVector<int> OriginalParameterIndex;
  
  for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if(F->getArg(i)->getNumUses() > 0) {
      OriginalParameterIndex.push_back(i);
    }
  }

  llvm::SmallVector<llvm::Type*, 8> NewParams;
  for(auto Index : OriginalParameterIndex) {
    NewParams.push_back(F->getFunctionType()->getParamType(Index));
  }

  llvm::FunctionType *NewFType = llvm::FunctionType::get(F->getReturnType(), NewParams, false);
  if (auto *NewF = llvm::dyn_cast<llvm::Function>(
        M.getOrInsertFunction(FunctionName, NewFType).getCallee())) {
    
    for(auto Attr : F->getAttributes().getFnAttrs())
      NewF->addFnAttr(Attr);
    for(int i = 0; i < OriginalParameterIndex.size(); ++i){
      for (auto Attr : F->getAttributes().getParamAttrs(OriginalParameterIndex[i]))
        NewF->addParamAttr(i, Attr);
    }
    NewF->setLinkage(F->getLinkage());
    NewF->setCallingConv(F->getCallingConv());

    // Now create function call to old function and inline
    llvm::BasicBlock *BB =
        llvm::BasicBlock::Create(M.getContext(), "", NewF);
    
    llvm::SmallVector<llvm::Value*> CallArgs;
    
    for(int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
      int NewFArgIndex = -1;
      for(int j = 0; j < OriginalParameterIndex.size(); ++j) {
        if(OriginalParameterIndex[j] == i) {
          NewFArgIndex = j;
          break;
        }
      }
      if(NewFArgIndex == -1)
        CallArgs.push_back(llvm::UndefValue::get(F->getFunctionType()->getParamType(i)));
      else
        CallArgs.push_back(NewF->getArg(NewFArgIndex));
    }
    
    auto *Call = llvm::CallInst::Create(llvm::FunctionCallee(F), CallArgs,
                                        "", BB);
    if(F->getReturnType()->isVoidTy())
      llvm::ReturnInst::Create(M.getContext(), BB);
    else
      llvm::ReturnInst::Create(M.getContext(), Call, BB);

    if(!F->hasFnAttribute(llvm::Attribute::AlwaysInline))
      F->addFnAttr(llvm::Attribute::AlwaysInline);
    F->setLinkage(llvm::GlobalValue::InternalLinkage);

    if(RetainedParameterIndices) {
      RetainedParameterIndices->clear();

      for(auto Index : OriginalParameterIndex)
        RetainedParameterIndices->push_back(Index);
    }
  }
}
}

DeadArgumentEliminationPass::DeadArgumentEliminationPass(
    llvm::Function* F, llvm::SmallVector<int>* RetainedArgs)
    : TargetFunction{F}, RetainedArguments{RetainedArgs} {}

llvm::PreservedAnalyses DeadArgumentEliminationPass::run(llvm::Module &M,
                                                        llvm::ModuleAnalysisManager &MAM) {
  
  if(TargetFunction)
    removeUnusedFunctionParameters(TargetFunction, M, RetainedArguments);
  return llvm::PreservedAnalyses::none();
}
}
}
