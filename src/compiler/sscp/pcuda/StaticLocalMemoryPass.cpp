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


#include "hipSYCL/compiler/sscp/pcuda/StaticLocalMemoryPass.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"
#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>
namespace hipsycl {
namespace compiler {


namespace {

static const char* LocalMemAnnotation = "acpp_local_memory";

llvm::GlobalVariable* copyGVToAS(unsigned AS, llvm::Module& M, llvm::GlobalVariable* GV) {

  assert(GV);

  std::string VarName {GV->getName()};
  GV->setName(VarName+".original");

  llvm::Constant* Initalizer = nullptr;
  
  if(GV->hasInitializer()) {
    Initalizer = GV->getInitializer();
  }

  llvm::GlobalVariable *NewVar =
      new llvm::GlobalVariable(M, GV->getValueType(), GV->isConstant(), GV->getLinkage(),
                               Initalizer, VarName, nullptr, GV->getThreadLocalMode(), AS);

  NewVar->setAlignment(GV->getAlign());

  llvm::Value *V = llvm::ConstantExpr::getPointerCast(NewVar, GV->getType());

  GV->replaceAllUsesWith(V);
  GV->eraseFromParent();

  return NewVar;
}

llvm::ConstantDataArray* getAnnotationGVString(llvm::Value* V) {
  if(auto* GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
    return getAnnotationGVString(GEP->getPointerOperand());
  } else if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
    auto *Initializer = GV->getInitializer();
    if (Initializer) {
      if (auto *CDA = llvm::dyn_cast<llvm::ConstantDataArray>(Initializer)) {
        return CDA;
      }
    }
  } else if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
    int Opcode = CE->getOpcode();
    if (Opcode == llvm::Instruction::GetElementPtr || Opcode == llvm::Instruction::AddrSpaceCast ||
        Opcode == llvm::Instruction::BitCast) {
      return getAnnotationGVString(CE->getOperand(0));
    }
  }
  return nullptr;
}

llvm::GlobalVariable* createLocalMemGV(llvm::Type* T, unsigned LocalAS, llvm::Function* ParentF) {
  static unsigned counter = 0;
  std::string VarName =
      "__acpp_local_mem_" + ParentF->getName().str() + "." + std::to_string(counter++);

  llvm::Module& M = *ParentF->getParent();
  llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
      M, T, false, llvm::GlobalValue::InternalLinkage, llvm::UndefValue::get(T), VarName, nullptr,
      llvm::GlobalValue::ThreadLocalMode::NotThreadLocal, LocalAS);

  NewVar->setAlignment(M.getDataLayout().getPrefTypeAlign(T));
  return NewVar;
}

}



StaticLocalMemoryPass::StaticLocalMemoryPass(unsigned LocalMemAddressSpace)
: LocalMemAS{LocalMemAddressSpace} {}

llvm::PreservedAnalyses StaticLocalMemoryPass::run(llvm::Module &M,
                                                   llvm::ModuleAnalysisManager &MAM) {

  llvm::SmallVector<llvm::GlobalVariable*> LocalMemGVs;
  utils::findGVsWithStringAnnotations(M, [&](llvm::GlobalVariable* GV, llvm::StringRef Annotation){
    if(Annotation.compare(LocalMemAnnotation)==0){
      LocalMemGVs.push_back(GV);
    }
  });

  for(auto* GV : LocalMemGVs) {
    copyGVToAS(LocalMemAS, M, GV);
  }

  llvm::SmallVector<llvm::Instruction*> AnnotationInstsToRemove;

  for(auto& F : M) {
    for(auto& BB : F) {
      for(auto& I : BB) {
        if(auto* CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
          llvm::Function* Callee = CB->getCalledFunction();
          if(Callee && llvmutils::starts_with(Callee->getName(), "llvm.var.annotation")) {
            if(CB->getCalledFunction()->getFunctionType()->getNumParams() >= 2) {
              llvm::Value* AnnotationArg = CB->getArgOperand(1);
              if(auto* DataArray = getAnnotationGVString(AnnotationArg)) {
                llvm::StringRef Annotation = DataArray->getAsCString();
                if(Annotation.compare(LocalMemAnnotation) == 0) {
                  llvm::Value* VarArg = CB->getArgOperand(0);
                  if(auto* BI = llvm::dyn_cast<llvm::BitCastInst>(VarArg)) {
                    VarArg = BI->getOperand(0);
                  }
                  if(auto* AI = llvm::dyn_cast<llvm::AllocaInst>(VarArg)) {
                    llvm::GlobalVariable* GV = createLocalMemGV(AI->getAllocatedType(), LocalMemAS, &F);
                    llvm::Value *CastToGenericAS = new llvm::AddrSpaceCastInst{GV, AI->getType(), "", llvmutils::makeInsertionPoint(AI)};
                    AI->replaceUsesWithIf(CastToGenericAS, [&](llvm::Use& U){
                      if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U.getUser())) {
                        auto* Callee = CB->getCalledFunction();
                        if (Callee && llvmutils::starts_with(Callee->getName(), "llvm.lifetime."))
                          return false;
                      }
                      return true;
                    });
                  }
                  AnnotationInstsToRemove.push_back(CB);
                }
              }
            } 
          }
        }
      }
    }
  }

  for(auto* I : AnnotationInstsToRemove) {
    I->replaceAllUsesWith(llvm::UndefValue::get(I->getType()));
    I->dropAllReferences();
    I->eraseFromParent();
  }
  return llvm::PreservedAnalyses::none();
}
}
}