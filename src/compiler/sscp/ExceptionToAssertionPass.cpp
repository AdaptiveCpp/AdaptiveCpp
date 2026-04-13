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
#include "hipSYCL/compiler/sscp/ExceptionToAssertionPass.hpp"
#include <llvm/IR/PassManager.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Support/Alignment.h>
#include <llvm/IR/IRBuilder.h>
#include <vector>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <limits> // std::numeric_limits<int>::max()

namespace hipsycl {
namespace compiler {


llvm::PreservedAnalyses ExceptionToAssertionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static const char* CXAThrow = "__cxa_throw";
  static const char* ACPPSSCPAssertFail = "__acpp_sscp_assert_fail";
  static const char* CXAAllocExc = "__cxa_allocate_exception";

  // declare __acpp_sscp_assert_fail
  // detect if C++ throws occur in source code
  llvm::Function* OldCXAThrow = M.getFunction(OldCXAThrow);

  // declare __acpp_sscp_assert_fail if not yet declared
  llvm::Type* llvmCharType;
  llvm::Function* ACPPSSCPAssertFailDeclaration;
  #if LLVM_VERSION_MAJOR >= 16
  llvmCharType = llvm::PointerType::get(M.getContext(), 0);
  #else
  llvmCharType = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), 0);
  #endif
  if(auto* F = M.getFunction(ACPPSSCPAssertFail)){
    ACPPSSCPAssertFailDeclaration=F;
  }
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

  //store all invokes and calls of __cxa_throw for later
  llvm::SmallVector<llvm::InvokeInst*> cxaThrowInvokes;
  llvm::SmallVector<llvm::CallInst*> cxaThrowCalls;
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &Inst : BB) {
        if(auto *Invoke = llvm::dyn_cast<llvm::InvokeInst>(&Inst)) {
          if (Invoke->getCalledFunction()) {
            if(Invoke->getCalledFunction()->getName() == CXAThrow) {
                //count to the invoke instructions
                cxaThrowInvokes.push_back(Invoke); 
              }
            }
          }
          if(auto *Call = llvm::dyn_cast<llvm::CallInst>(&Inst)) {
          if (Call->getCalledFunction()) {
            if(Call->getCalledFunction()->getName() == CXAThrow) {
                //count to the call instructions
                cxaThrowCalls.push_back(Call);
              }
            }
          }
        }
    }
  }
  
  // create calls to __acpp_sscp_assert_fail and terminate with unreachable instructions in place of __cxa_throw invokes
  llvm::SmallVector<llvm::UnreachableInst*> UnreachableForCXAInvokes;
  for (int i=0; i<cxaThrowInvokes.size(); i++) {
    // construct device assertion signature
    const char *assertionStr = "Exception in Device Code";
    const char *fileStr = M.getSourceFileName().c_str();
    int lineNumber = std::numeric_limits<int>::max();
    if (llvm::DebugLoc DL = cxaThrowInvokes[i]->getDebugLoc()) { // if -g flag was used grab the line number
        lineNumber = DL.getLine();
    }
    std::string functionNameStr = cxaThrowInvokes[i]->getParent()->getParent()->getName().str();
    const char *functionName = functionNameStr.c_str(); 
    llvm::IRBuilder<> builder(cxaThrowInvokes[i]);

    llvm::SmallVector<llvm::Value*> ACPPSSCPAssertFailArgs;
    ACPPSSCPAssertFailArgs.push_back(builder.CreateGlobalStringPtr(assertionStr));
    ACPPSSCPAssertFailArgs.push_back(builder.CreateGlobalStringPtr(fileStr));
    ACPPSSCPAssertFailArgs.push_back(builder.getInt32(lineNumber));
    ACPPSSCPAssertFailArgs.push_back(builder.CreateGlobalStringPtr(functionName));
  
    auto *Call = llvm::CallInst::Create(ACPPSSCPAssertFailDeclaration, ACPPSSCPAssertFailArgs);
    Call->insertBefore(cxaThrowInvokes[i]);
    auto *Unreachable = new llvm::UnreachableInst(M.getContext(), cxaThrowInvokes[i]); 
    UnreachableForCXAInvokes.push_back(Unreachable);
  }
  // replace all uses of invoke @__cxa_throw with unreachable instruction and erase the invoke instruction
  for (int i=0; i<cxaThrowInvokes.size(); i++) {
    UnreachableForCXAInvokes[i]->replaceAllUsesWith(cxaThrowInvokes[i]);
    cxaThrowInvokes[i]->eraseFromParent();
  }

  // create calls to __acpp_sscp_assert_fail in place of __cxa_throw calls
  llvm::SmallVector<llvm::CallInst*> AssertForCXACalls;
  for (int i=0; i<cxaThrowCalls.size(); i++) {
    // construct device assertion signature
    const char *assertionStr = "Exception in Device Code";
    const char *fileStr = M.getSourceFileName().c_str();
    int lineNumber = -42;
    if (llvm::DebugLoc DL = cxaThrowCalls[i]->getDebugLoc()) { // if -g flag was used grab the line number
        lineNumber = DL.getLine();
    }

    std::string functionNameStr = cxaThrowCalls[i]->getParent()->getParent()->getName().str();
    const char *functionName = functionNameStr.c_str(); 
    llvm::IRBuilder<> builder(cxaThrowCalls[i]);

    llvm::SmallVector<llvm::Value*> ACPPSSCPAssertFailArgs;
    ACPPSSCPAssertFailArgs.push_back(builder.CreateGlobalStringPtr(assertionStr));
    ACPPSSCPAssertFailArgs.push_back(builder.CreateGlobalStringPtr(fileStr));
    ACPPSSCPAssertFailArgs.push_back(builder.getInt32(lineNumber));
    ACPPSSCPAssertFailArgs.push_back(builder.CreateGlobalStringPtr(functionName));
  
    auto *Call = llvm::CallInst::Create(ACPPSSCPAssertFailDeclaration, ACPPSSCPAssertFailArgs);
    Call->insertBefore(cxaThrowCalls[i]);
    AssertForCXACalls.push_back(Call);
  }
  // replace all uses of call @__cxa_throw with call to __acpp_sscp_assert_fail and clean up
  for (int i=0; i<cxaThrowCalls.size(); i++) {
    AssertForCXACalls[i]->replaceAllUsesWith(cxaThrowCalls[i]);
    cxaThrowCalls[i]->eraseFromParent();
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

  return llvm::PreservedAnalyses::none();
}



}

}



