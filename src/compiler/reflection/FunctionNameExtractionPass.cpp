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
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <string>
#include "hipSYCL/compiler/reflection/FunctionNameExtractionPass.hpp"
#include "hipSYCL/compiler/utils/ProcessFunctionAnnotationsPass.hpp"
#include <llvm/IR/PassManager.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Support/Alignment.h>



namespace hipsycl {
namespace compiler {

namespace {

constexpr const char AnnotationName [] = "needs_function_ptr_argument_reflection";

bool isReflectionAnnotatedFunction(llvm::Function* F, const utils::ProcessFunctionAnnotationPass& PFA) {
  auto It = PFA.getFoundAnnotations().find(AnnotationName);
  if(It == PFA.getFoundAnnotations().end())
    return false;

  for(auto* Candidate : It->second) {
    if(Candidate == F) {
      return true;
    }
  }

  return false;
}

bool isAnyUserReflectionAnnotatedFunction(llvm::Function* F, const utils::ProcessFunctionAnnotationPass& PFA) {
  for(auto* U : F->users()) {
    if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U)) {
      auto* CalledF = CB->getCalledFunction();
      if(CalledF && (CalledF != F)) {
      
        if(isReflectionAnnotatedFunction(CalledF, PFA))
          return true;
      
      }
    }
  }
  return false;
}

}

llvm::Type* getCharPtrType(llvm::Module* M, unsigned AS = 0) {
  #if LLVM_VERSION_MAJOR >= 16
      return llvm::PointerType::get(M->getContext(), AS);
  #else
      return llvm::PointerType::get(llvm::Type::getInt8Ty(M->getContext()), AS);
  #endif
  }
  llvm::Type* getVoidPtrType(llvm::Module* M, unsigned AS = 0) {
    #if LLVM_VERSION_MAJOR >= 16
        return llvm::PointerType::get(M->getContext(), AS);
    #else
        return llvm::PointerType::get(llvm::Type::getInt8Ty(M->getContext()), AS);
    #endif
  }

llvm::PreservedAnalyses FunctionNameExtractionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  std::string GVPrefix = "__acpp_functioninfo_";

  utils::ProcessFunctionAnnotationPass PFA{{AnnotationName}};
  PFA.run(M, AM);

  llvm::SmallVector<std::pair<llvm::Function*, llvm::GlobalVariable*>> FunctionsToProcess;
  for(auto& F : M) {
    if(F.hasAddressTaken() && isAnyUserReflectionAnnotatedFunction(&F, PFA)) {
      auto FunctionName = F.getName().str();
      llvm::Constant *Initializer = llvm::ConstantDataArray::getRaw(
          FunctionName + '\0', FunctionName.size() + 1, llvm::Type::getInt8Ty(M.getContext()));
      
      llvm::GlobalVariable *NameGV = new llvm::GlobalVariable(M, Initializer->getType(), true,
                                                                llvm::GlobalValue::InternalLinkage,
                                                                Initializer, GVPrefix+FunctionName);
      NameGV->setAlignment(llvm::Align{1});
      FunctionsToProcess.push_back(std::make_pair(&F, NameGV));
    }
  }

  //declare explicitly __acpp_reflection_associate_function_pointer
  static const char* ReflectionAssociateFunctionPointer = "__acpp_reflection_associate_function_pointer";
  llvm::SmallVector<llvm::Type*> ParamTs;
  // function pointer
  ParamTs.push_back(getVoidPtrType(&M));
  // function name
  ParamTs.push_back(getCharPtrType(&M));  
  // declare if not already declared
  auto FC = M.getOrInsertFunction(ReflectionAssociateFunctionPointer,
                                  llvm::FunctionType::get(llvm::Type::getVoidTy(M.getContext()),
                                                          llvm::ArrayRef<llvm::Type *>{ParamTs},
                                                          false));
  llvm::Function *NewDeclaration = llvm::dyn_cast<llvm::Function>(FC.getCallee());
  std::string NewDeclarationName = NewDeclaration->getName().str();
  
  
  llvm::Function* MapFunc = nullptr;
  for(auto& F : M)
    if(F.getName().contains("__acpp_reflection_associate_function_pointer"))
      MapFunc = &F;
  
  if(MapFunc) {
    if(auto* InitFunc = M.getFunction("__acpp_reflection_init_registered_function_pointers")){
      if(InitFunc->isDeclaration() && (MapFunc->getFunctionType()->getNumParams() == 2)) {
        InitFunc->setLinkage(llvm::GlobalValue::InternalLinkage);

        llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", InitFunc);

        for(const auto& FuncGVPair : FunctionsToProcess) {
          llvm::SmallVector<llvm::Value*> Args;
          Args.push_back(llvm::BitCastInst::Create(llvm::Instruction::BitCast, FuncGVPair.first,
                                                   MapFunc->getFunctionType()->getParamType(0), "",
                                                   BB));
          
          auto Zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 0);
          llvm::SmallVector<llvm::Value*> GEPIndices{Zero, Zero};
          auto *GVGEPInst = llvm::GetElementPtrInst::CreateInBounds(
              FuncGVPair.second->getValueType(), FuncGVPair.second,
              llvm::ArrayRef<llvm::Value *>{GEPIndices}, "", BB);
          Args.push_back(GVGEPInst);
              
          llvm::CallInst::Create(llvm::FunctionCallee(MapFunc), llvm::ArrayRef<llvm::Value *>{Args},
                                 "", BB);
        }

        llvm::ReturnInst::Create(M.getContext(), BB);
      }
    }
  }

  return llvm::PreservedAnalyses::none();
}


}
}

