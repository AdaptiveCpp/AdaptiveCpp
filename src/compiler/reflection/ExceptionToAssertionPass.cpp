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


  // declare __acpp_sscp_assert_fail
  // detect if C++ throws occur in source code
  llvm::Function* oldcxaThrow = nullptr;
  for(auto& F : M)
    if(F.getName().contains(CXAThrow)) {
      oldcxaThrow = &F;
    }
  if (!oldcxaThrow) {
    llvm::errs() << "Function " << CXAThrow << " not found. Assuming there is no use of exceptions in source code.\n";
    return llvm::PreservedAnalyses::none();
  }

  // declare __acpp_sscp_assert_fail if not yet declared
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

  //store all invokes and calls of __cxa_throw and their args for later RAUW
  llvm::SmallVector<llvm::InvokeInst*> cxaThrowInvokes;
  llvm::SmallVector<llvm::SmallVector<llvm::Value*>> cxaThrowInvokesArgs;
  llvm::SmallVector<llvm::CallInst*> cxaThrowCalls;
  llvm::SmallVector<llvm::SmallVector<llvm::Value*>> cxaThrowCallsArgs;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &Inst : BB) {
        if(auto *Invoke = llvm::dyn_cast<llvm::InvokeInst>(&Inst)) {
          if (Invoke->getCalledFunction()) { //invoke of fct pointer would return nullptr for getCalledFunction()
            if(Invoke->getCalledFunction()->getName() == CXAThrow) {
                //count to the invoke instructions
                cxaThrowInvokes.push_back(Invoke); 
                //retrieve each __cxa_throw-use's arguments
                llvm::User::op_iterator cxaThrowArgIt;
                llvm::SmallVector<llvm::Value*> cxaThrowArgs;

                for (cxaThrowArgIt=Invoke->arg_begin(); cxaThrowArgIt != Invoke->arg_end(); cxaThrowArgIt++) {
                  cxaThrowArgs.push_back(*cxaThrowArgIt);
                }
                cxaThrowInvokesArgs.push_back(cxaThrowArgs);
              }
            }
          }
          if(auto *Call = llvm::dyn_cast<llvm::CallInst>(&Inst)) {
          if (Call->getCalledFunction()) { //Call of fct pointer would return nullptr for getCalledFunction()
            if(Call->getCalledFunction()->getName() == CXAThrow) {
                //count to the call instructions
                cxaThrowCalls.push_back(Call); 
                //retrieve each __cxa_throw-use's arguments
                llvm::User::op_iterator cxaThrowArgIt;
                llvm::SmallVector<llvm::Value*> cxaThrowArgs;

                for (cxaThrowArgIt=Call->arg_begin(); cxaThrowArgIt != Call->arg_end(); cxaThrowArgIt++) {
                  cxaThrowArgs.push_back(*cxaThrowArgIt);
                }
                cxaThrowCallsArgs.push_back(cxaThrowArgs);
              }
            }
          }
        }
    }
  }


  //build new call to __acpp_sscp_assert_fail
  for (int i=0; i<cxaThrowInvokes.size(); i++) {
    auto *Invoke = cxaThrowInvokes[i];
    llvm::SmallVector<llvm::Value*> InvokeArgs = cxaThrowInvokesArgs[i];
    llvm::IRBuilder<> Builder(Invoke);
    llvm::CallInst *newCall = Builder.CreateCall(ACPPSSCPAssertFailDeclaration, InvokeArgs);
    newCall->setCallingConv(Invoke->getCallingConv());
    newCall->setDebugLoc(Invoke->getDebugLoc());
    //Invoke->replaceAllUsesWith(newCall);
    llvm::Instruction *terminator = Builder.CreateUnreachable();

  }
  for (int i=0; i<cxaThrowCalls.size(); i++) {
    auto *Call = cxaThrowCalls[i];
    llvm::SmallVector<llvm::Value*> CallArgs = cxaThrowCallsArgs[i];
    llvm::IRBuilder<> Builder(Call);
    llvm::CallInst *newCall = Builder.CreateCall(ACPPSSCPAssertFailDeclaration, CallArgs);
    newCall->setCallingConv(Call->getCallingConv());
    newCall->setDebugLoc(Call->getDebugLoc());
    //Call->replaceAllUsesWith(newCall);
    llvm::Instruction *terminator = Builder.CreateUnreachable();
  }    

  //remove all invokes and calls to __cxa_throw
  for (auto *Invoke : cxaThrowInvokes) {
    //remove the __cxa_throw invoke
    Invoke->eraseFromParent();
  }
  for (auto *Call : cxaThrowCalls) {
    //remove the __cxa_throw call
    Call->eraseFromParent();
  }
  


  //throws in C++ mean memory allocation of exception structure by __cxa_allocate_exception
  //we replace these calls by alloca instructions

  //find all calls to __cxa_allocate_exception and RAUW alloca instructions
  llvm::SmallVector<llvm::CallInst*> cxaAllocExcCalls;
  llvm::SmallVector<llvm::SmallVector<llvm::Value*>> cxaAllocExcCallsArgs;  
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if(auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
          if (Call->getCalledFunction()) {
            if(Call->getCalledFunction()->getName() == CXAAllocExc) {
              // count to the __cxa_allocate_exception calls
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
              cxaAllocExcCallsArgs.push_back(cxaAllocExcArgs);


              // construct the alloca instruction
              // allocation size to be allocated is first arg in __cxa_allocate_exception signature
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
              Call->replaceAllUsesWith(allocainstruction); //RAUW
            }
          }
        }
      }
    }
  }

  // reiterate to get updated pointers to all calls to __cxa_allocate_exception
  llvm::SmallVector<llvm::CallInst*> cxaAllocExcCalls2;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if(auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
          if (Call->getCalledFunction()) {
            if(Call->getCalledFunction()->getName() == CXAAllocExc) {
              // count to the call instructions
              cxaAllocExcCalls2.push_back(Call); 
            }
          }
        }
      }
    }
  }

  //remove all calls to __cxa_allocate_exception
  for (auto *Call : cxaAllocExcCalls2) {
      if(Call!=nullptr)
      Call->eraseFromParent();
  }

  M.print(llvm::errs(), nullptr);
  return llvm::PreservedAnalyses::none();
}



}

}

