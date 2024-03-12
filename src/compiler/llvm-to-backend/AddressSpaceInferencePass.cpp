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

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Value.h>
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
#include "hipSYCL/compiler/llvm-to-backend/AddressSpaceMap.hpp"

namespace hipsycl {
namespace compiler {

namespace {


llvm::GlobalVariable *setGlobalVariableAddressSpace(llvm::Module &M, llvm::GlobalVariable *GV,
                                                    unsigned AS) {
  assert(GV);

  std::string VarName {GV->getName()};
  GV->setName(VarName+".original");

  llvm::Constant* Initalizer = nullptr;
  
  if(GV->hasInitializer()) {
    Initalizer = GV->getInitializer();
  }

  llvm::GlobalVariable* NewVar = new llvm::GlobalVariable(
      M, GV->getValueType(), GV->isConstant(), GV->getLinkage(), Initalizer, VarName, nullptr,
      GV->getThreadLocalMode(), AS);
  
  NewVar->setAlignment(GV->getAlign());
  llvm::Value *V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());

  GV->replaceAllUsesWith(V);
  GV->eraseFromParent();

  return NewVar;
}

// Go through all users, but look through addrspacecasts, bitcasts and getelementptr
template<class F>
void forEachUseOfPointerValue(llvm::Value* V, F&& handler) {
  for(llvm::Value* U : V->users()) {
    if (llvm::isa<llvm::BitCastInst>(U) || llvm::isa<llvm::AddrSpaceCastInst>(U) ||
        llvm::isa<llvm::GetElementPtrInst>(U)) {
      forEachUseOfPointerValue(U, handler);
    } else {
      handler(U);
    }
  }
}

} // anonymous namespace


AddressSpaceInferencePass::AddressSpaceInferencePass(const AddressSpaceMap &Map) : ASMap{Map} {}

llvm::PreservedAnalyses AddressSpaceInferencePass::run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
  
  if(ASMap[AddressSpace::Generic] != 0){
    HIPSYCL_DEBUG_ERROR << "AddressSpaceInferencePass: Attempted to run when default address space "
                           "is not generic address space. This is not yet supported.\n";
  }

  assert(ASMap[AddressSpace::Generic] == 0);

  // Fix global vars
  llvm::SmallVector<std::pair<llvm::GlobalVariable *, unsigned>> GlobalVarAddressSpaceChanges;
  for(auto& G : M.globals()) {
    unsigned CurrentAS = G.getAddressSpace();
    // By default, all global vars should go into global var default AS
    unsigned TargetAS = ASMap[AddressSpace::GlobalVariableDefault];

    if (CurrentAS == ASMap[AddressSpace::Local]) {
      // Don't do anything for explicitly local global variables
      TargetAS = CurrentAS;
    } else if (G.isConstant()) {
      // constants go into AS for constant globals
      TargetAS = ASMap[AddressSpace::ConstantGlobalVariableDefault];
    }
    if(TargetAS != CurrentAS)
      GlobalVarAddressSpaceChanges.push_back(std::make_pair(&G, TargetAS));
  }
  for(auto& G : GlobalVarAddressSpaceChanges)
    setGlobalVariableAddressSpace(M, G.first, G.second);

  // If the target data layout has changed default alloca address space
  // we can end up with allocas that are in the wrong address space. We
  // need to fix this now.
  unsigned AllocaAddrSpace = ASMap[AddressSpace::AllocaDefault];
  llvm::SmallVector<llvm::Instruction*, 16> InstsToRemove;
  for(auto& F : M) {
    for(auto& BB : F) {
      for(auto& I : BB) {
        if(auto* AI = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
          if(AI->getAddressSpace() != AllocaAddrSpace) {
            HIPSYCL_DEBUG_INFO << "AddressSpaceInferencePass: Found alloca in address space "
                               << AI->getAddressSpace() << " when it should be in AS "
                               << AllocaAddrSpace << ", fixing.\n";
            auto *NewAI = new llvm::AllocaInst{AI->getAllocatedType(), AllocaAddrSpace, "", AI};
            auto* ASCastInst = new llvm::AddrSpaceCastInst{NewAI, AI->getType(), "", AI};

            // llvm.lifetime intrinsics don't like addrspacecasts,
            // so we cannot just make them use ASCastInst instead of AI now.
            forEachUseOfPointerValue(AI, [&](llvm::Value* U){
              if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U)) {
                llvm::StringRef CalleeName = CB->getCalledFunction()->getName();
                if(CalleeName.startswith("llvm.lifetime")) {
                  InstsToRemove.push_back(CB);

                  llvm::Intrinsic::ID Id = CalleeName.startswith("llvm.lifetime.start")
                                               ? llvm::Intrinsic::lifetime_start
                                               : llvm::Intrinsic::lifetime_end;

                  llvm::SmallVector<llvm::Type*> IntrinsicType {NewAI->getType()};
                  llvm::Function *LifetimeIntrinsic =
                      llvm::Intrinsic::getDeclaration(&M, Id, IntrinsicType);
                  llvm::SmallVector<llvm::Value*> CallArgs{CB->getArgOperand(0), NewAI};
                  llvm::CallInst::Create(llvm::FunctionCallee(LifetimeIntrinsic), CallArgs, "", CB);
                }
              }
            });

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
