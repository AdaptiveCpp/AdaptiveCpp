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
#include "hipSYCL/compiler/reflection/ExceptionToAssertionPass.hpp"
#include <llvm/IR/PassManager.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Support/Alignment.h>
#include <llvm/IR/IRBuilder.h>
#include <vector>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace hipsycl {
namespace compiler {

llvm::PreservedAnalyses ExceptionToAssertionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  //Strategy: __cxa__throw retains its old signature but internally only calls __acpp_sscp_assert_fail 

  static const char* CXAThrow = "__cxa_throw";
  static const char* ACPPSSCPAssertFail = "__acpp_sscp_assert_fail";
  //static const char* GlibcxxAssertFailBuiltinName = "__acpp_sscp_glibcxx_assert_fail";
  static const char* CXAAllocExc = "__cxa_allocate_exception";
  
  
  // CloneFunction and change name
  // fetch
  llvm::Function* oldcxaThrow = nullptr;
  for(auto& F : M)
    if(F.getName().contains(CXAThrow)) {
      oldcxaThrow = &F;
    }
  if (!oldcxaThrow) {
    llvm::errs() << "Function __cxa_throw" << CXAThrow << " not found. Assuming there is no use of exceptions in source code.\n";
    return llvm::PreservedAnalyses::none();
  }
  
  llvm::Type* llvmCharType;
  llvm::Function* ACPPSSCPAssertFailDeclaration;
  #if LLVM_VERSION_MAJOR >= 16
      llvmCharType = llvm::PointerType::get(M.getContext(), 0);
  #else
      llvmCharType = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), 0);
  #endif
  if(auto* F = M.getFunction(ACPPSSCPAssertFail))
    ACPPSSCPAssertFailDeclaration=F;
  else {
    llvm::SmallVector<llvm::Type*> ParamTs;
    // assertion
    ParamTs.push_back(llvmCharType);
    // file
    ParamTs.push_back(llvmCharType);
    // line
    ParamTs.push_back(llvm::Type::getInt32Ty(M.getContext()));
    // function name
    ParamTs.push_back(llvmCharType);

    auto FC = M.getOrInsertFunction(ACPPSSCPAssertFail,
                                    llvm::FunctionType::get(llvm::Type::getVoidTy(M.getContext()),
                                                            llvm::ArrayRef<llvm::Type *>{ParamTs},
                                                            false));
    ACPPSSCPAssertFailDeclaration = llvm::dyn_cast<llvm::Function>(FC.getCallee());
    ACPPSSCPAssertFailDeclaration->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
  }

  //store all invokes of __cxa_throw for later removal
  llvm::SmallVector<llvm::InvokeInst*> cxaThrowInvokes;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if(auto *Invoke = llvm::dyn_cast<llvm::InvokeInst>(&I)) {
          if (Invoke->getCalledFunction()->getName() == CXAThrow) {
              llvm::IRBuilder<> Builder(Invoke);
              //count to the invoke instructions
              cxaThrowInvokes.push_back(Invoke); 
              //retrieve each __cxa_throw-use's arguments
              llvm::User::op_iterator cxaThrowArgIt;
              llvm::SmallVector<llvm::Value*> cxaThrowArgs;
              for (cxaThrowArgIt=Invoke->arg_begin(); cxaThrowArgIt != Invoke->arg_end(); cxaThrowArgIt++) {
                cxaThrowArgs.push_back(*cxaThrowArgIt);
              }
              llvm::CallInst *newCall = Builder.CreateCall(ACPPSSCPAssertFailDeclaration, llvm::ArrayRef<llvm::Value*>{cxaThrowArgs});
              newCall->setCallingConv(Invoke->getCallingConv());
              newCall->setDebugLoc(Invoke->getDebugLoc());
              Invoke->replaceAllUsesWith(newCall);
              llvm::Instruction *terminator = Builder.CreateUnreachable();
            }
          }
      }
    }
  }
  for (auto *Invoke : cxaThrowInvokes) {
    //remove the invoke instruction
    Invoke->eraseFromParent();
  }


  //remove all calls to __cxa_allocate_exception
  llvm::SmallVector<llvm::CallInst*> cxaAllocExcCalls;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if(auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
          if (Call->getCalledFunction()) {
            if(Call->getCalledFunction()->getName() == CXAAllocExc) {
              // count to the invoke instructions
              cxaAllocExcCalls.push_back(Call); 
              // retain amount of data to be allocated
              // Assuming always first argument/llvm::Value is alloc size of the exception object AND that it is specified as i64
              llvm::User::op_iterator cxaAllocExcIt;
              llvm::SmallVector<llvm::Value*> cxaAllocExcArgs;
              int i = 0;
              for (cxaAllocExcIt=Call->arg_begin(); cxaAllocExcIt != Call->arg_end(); cxaAllocExcIt++) {
                cxaAllocExcArgs.push_back(*cxaAllocExcIt);
                i++;
              }

              // construct the alloca instruction
              llvm::Value *allocSize;
              if (auto *ConstInt = llvm::dyn_cast<llvm::ConstantInt>(cxaAllocExcArgs[0])) {
                // Check if the type is i64
                if (ConstInt->getType()->isIntegerTy(64)) {
                  // Retrieve the zero-extended value and create array size value
                  allocSize = llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 
                                                                  static_cast<int>(ConstInt->getZExtValue()));
                }
              }
              auto* allocainstruction = new llvm::AllocaInst(cxaAllocExcArgs[0]->getType(), 0, allocSize, "exception_alloca", Call);
              Call->replaceAllUsesWith(allocainstruction);
            }
          }
        }
      }
    }
  }
  llvm::SmallVector<llvm::CallInst*> cxaAllocExcCalls2;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if(auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
          if (Call->getCalledFunction()) {
            if(Call->getCalledFunction()->getName() == CXAAllocExc) {
              // count to the invoke instructions
              cxaAllocExcCalls2.push_back(Call); 
            }
          }
        }
      }
    }
  }
  for (auto *Call : cxaAllocExcCalls2) {
    // remove the call of cxa_allocate_exception
    if(Call!=nullptr)
      Call->eraseFromParent();
  }
  return llvm::PreservedAnalyses::none();
}



}

}

