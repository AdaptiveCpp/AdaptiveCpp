/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#include <cassert>
#include <llvm/IR/BasicBlock.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Constants.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceInferencePass.hpp"

namespace hipsycl {
namespace compiler {

namespace {

AddressSpace getTargetAddressSpace(const llvm::GlobalVariable *GV,
                                   const AddressSpaceMap &PlaceholderASMap) {
  assert(GV);

  if(GV->getAddressSpace() == PlaceholderASMap[AddressSpace::Constant])
    return AddressSpace::Constant;
  else if(GV->getAddressSpace() == PlaceholderASMap[AddressSpace::Global])
    return AddressSpace::Global;
  else if(GV->getAddressSpace() == PlaceholderASMap[AddressSpace::Local])
    return AddressSpace::Local;
  else if(GV->getAddressSpace() == PlaceholderASMap[AddressSpace::Generic]) {
    HIPSYCL_DEBUG_WARNING << "Encountered global variable with explicit generic address space; "
                             "this is unsupported. Ignoring.\n";
    return AddressSpace::Global;
  } else if(GV->getAddressSpace() == PlaceholderASMap[AddressSpace::Private]) {
    HIPSYCL_DEBUG_WARNING << "Encountered global variable with explicit private address space; "
                             "this is unsupported. Ignoring.\n";
    return AddressSpace::Global;
  }

  if(GV->isConstant())
    return AddressSpace::Constant;
  else
    return AddressSpace::Global;
}

llvm::GlobalVariable *setGlobalVariableAddressSpace(llvm::Module &M, llvm::GlobalVariable *GV,
                                                    unsigned AS) {
  assert(GV);

  std::string VarName {GV->getName()};
  GV->setName(VarName+".original");

  llvm::Type *NewType = llvm::PointerType::getWithSamePointeeType(GV->getType(), AS);
  llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
      M, NewType, GV->isConstant(), GV->getLinkage(), GV->getInitializer(), VarName);
  
  llvm::Value* V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());
  GV->replaceAllUsesWith(V);
  GV->eraseFromParent();

  return NewVar;
}


} // anonymous namespace

void rewriteKernelArgumentAddressSpacesTo(unsigned AddressSpace, llvm::Module &M,
                                          const std::vector<std::string> &KernelNames,
                                          PassHandler &PH) {

  for(auto KernelName : KernelNames) {
    if(auto* F = M.getFunction(KernelName)){
      F->setName(KernelName+"_entrypoint");
      F->setLinkage(llvm::GlobalValue::InternalLinkage);

      llvm::SmallVector<llvm::Type*, 8> Params;
      llvm::FunctionType* OriginalFType = F->getFunctionType();
      
      for(int i = 0; i < OriginalFType->getNumParams(); ++i){
        llvm::Type* CurrentParamType = OriginalFType->getParamType(i);
        if(llvm::PointerType* PT = llvm::dyn_cast<llvm::PointerType>(CurrentParamType)) {
          if(PT->getAddressSpace() != AddressSpace) {
            llvm::Type *NewT =
                llvm::PointerType::getWithSamePointeeType(PT, AddressSpace);
            Params.push_back(NewT);
          }
        } else {
          Params.push_back(CurrentParamType);
        }
      }

      llvm::FunctionType *FType = llvm::FunctionType::get(F->getReturnType(), Params, false);

      if (auto *NewF = llvm::dyn_cast<llvm::Function>(
              M.getOrInsertFunction(KernelName, FType, F->getAttributes()).getCallee())) {

        llvm::BasicBlock *BB =
            llvm::BasicBlock::Create(M.getContext(), "" + KernelName, NewF);
        llvm::SmallVector<llvm::Value*, 8> CallArgs;
        for(int i = 0; i < Params.size(); ++i) {
          if(llvm::PointerType* NewPT = llvm::dyn_cast<llvm::PointerType>(NewF->getArg(i)->getType())) {
            
            assert(llvm::isa<llvm::PointerType>(F->getArg(i)->getType()));
            llvm::PointerType* OldPT = llvm::dyn_cast<llvm::PointerType>(F->getArg(i)->getType());
            
            if(NewPT->getAddressSpace() != OldPT->getAddressSpace()) {

              auto *ASCastInst = new llvm::AddrSpaceCastInst{NewF->getArg(i), OldPT,
                                                             "", BB};
              CallArgs.push_back(ASCastInst);
            } else {
              CallArgs.push_back(NewF->getArg(i));
            }
          } else {
            CallArgs.push_back(NewF->getArg(i));
          }
        }

        assert(CallArgs.size() == F->getFunctionType()->getNumParams());
        for(int i = 0; i < CallArgs.size(); ++i) {
          assert(CallArgs[i]->getType() == F->getFunctionType()->getParamType(i));
        }

        auto *Call = llvm::CallInst::Create(llvm::FunctionCallee(F), CallArgs,
                                            "", BB);
        llvm::ReturnInst::Create(M.getContext(), BB);
      }
    }
  }

  PH.ModuleAnalysisManager->invalidate(M, llvm::PreservedAnalyses::none());
}

AddressSpaceInferencePass::AddressSpaceInferencePass(const AddressSpaceMap &Map) : ASMap{Map} {}

llvm::PreservedAnalyses AddressSpaceInferencePass::run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {

  // TODO Set address space of global variables
  
  if(ASMap[AddressSpace::Generic] != 0){
    HIPSYCL_DEBUG_ERROR << "AddressSpaceInferencePass: Attempted to run when default address space "
                           "is not generic address space. This is not yet supported.\n";
  }

  assert(ASMap[AddressSpace::Generic] == 0);

  // If the target data layout has changed default alloca address space
  // we can end up with allocas that are in the wrong address space. We
  // need to fix this now.
  unsigned AllocaAddrSpace = ASMap[AddressSpace::AllocaDefault];
  llvm::SmallVector<llvm::Instruction*, 16> InstsToRemove;
  for(auto& F : M.getFunctionList()) {
    for(auto& BB : F.getBasicBlockList()) {
      for(auto& I : BB.getInstList()) {
        if(auto* AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
          if(AI->getAddressSpace() != AllocaAddrSpace) {
            auto *NewAI = new llvm::AllocaInst{AI->getAllocatedType(), AllocaAddrSpace, "", AI};
            auto* ASCastInst = new llvm::AddrSpaceCastInst{NewAI, AI->getType(), "", AI};
            AI->replaceAllUsesWith(ASCastInst);
            InstsToRemove.push_back(AI);
          }
        }
      }
    }
  }
  for(auto* I : InstsToRemove)
    I->eraseFromParent();
  
  auto IAS = llvm::createModuleToFunctionPassAdaptor(
      llvm::InferAddressSpacesPass{ASMap[AddressSpace::Generic]});
  IAS.run(M, MAM);

  return llvm::PreservedAnalyses::none();
}

}
}
