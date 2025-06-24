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

namespace hipsycl {
namespace compiler {

llvm::PreservedAnalyses ExceptionToAssertionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  //Strategy: Duplicate landing pad/unwind destination block but add in a call to __acpp_sscp_assert_fail in the exception path
  
  static const char* CXAThrow = "__cxa_throw";
  static const char* ACPPSSCPAssertFail = "__acpp_sscp_assert_fail";
  //static const char* GlibcxxAssertFailBuiltinName = "__acpp_sscp_glibcxx_assert_fail";
  
  //get old LP
  for (auto &F : M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if(auto *Invoke = llvm::dyn_cast<llvm::InvokeInst>(&I)) {
          if (Invoke->getCalledFunction()->getName() == CXAThrow) {
            //get landingpad instruction in old unwind block
            llvm::BasicBlock* oldUnwindBB = Invoke->getUnwindDest();
            llvm::LandingPadInst* oldLP = llvm::dyn_cast<llvm::LandingPadInst>(oldUnwindBB->getFirstNonPHI());
            if (!oldLP) {
              llvm::errs() << "Expected a landingpad inst in unwind block of __cxa_throw function!\n";
              continue;
            }
            //create new unwind block with duplicate landing pad
            llvm::BasicBlock* newLPadBB = llvm::BasicBlock::Create(M.getContext(), "cloned_lpad", &F);
            llvm::IRBuilder<> newLPadBBBuilder(newLPadBB);
            auto *newLP = newLPadBBBuilder.CreateLandingPad(oldLP->getType(), oldLP->getNumClauses(), "cloned_lp");
            newLP->setCleanup(oldLP->isCleanup());

            for (unsigned i = 0; i < oldLP->getNumClauses(); ++i) {
              newLP->addClause(oldLP->getClause(i));
            }

            //insert a call to the assertion failure function, later the DeviceAssertPass will replace this with the actual assertion failure function
            //reproduce function signature variables of __acpp_sscp_assert_fail
            #if LLVM_VERSION_MAJOR >= 16
              llvm::Type *charTy = llvm::PointerType::get(M.getContext(), 0);
              llvm::Type *voidTy = llvm::PointerType::get(M.getContext(), 0);
            #else
              llvm::Type *charTy = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), 0); //assuming chartype ptr is same data width as int8 ptr
              llvm::Type *voidTy = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), 0);
            #endif        
            llvm::Type *i32Ty = llvm::PointerType::get(llvm::Type::getInt32Ty(M.getContext()), 0);
            
            //reproduce function type of function __acpp_sscp_assert_fail
            llvm::SmallVector<llvm::Type*> ParamTs;
            ParamTs.push_back(charTy);
            ParamTs.push_back(charTy);
            ParamTs.push_back(i32Ty);
            ParamTs.push_back(charTy);
            llvm::FunctionType *ACPPSSCPAssertFailTy = llvm::FunctionType::get(voidTy, llvm::ArrayRef<llvm::Type *>{ParamTs}, false);

            //declare __acpp_sscp_assert_fail explicitly
            auto FC = M.getOrInsertFunction(ACPPSSCPAssertFail, ACPPSSCPAssertFailTy);
            llvm::Function *ACPPSSCPAssertFailDeclaration = llvm::dyn_cast<llvm::Function>(FC.getCallee());
            ACPPSSCPAssertFailDeclaration->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);

            //create the call to (presumably empty) __acpp_sscp_assert_fail declaration
            llvm::Value* AssertionStr = newLPadBBBuilder.CreateGlobalStringPtr("Exception caught, replacing with call to (presumably empty) __acpp_sscp_assert_fail declaration");
            newLPadBBBuilder.CreateCall(ACPPSSCPAssertFailDeclaration, {AssertionStr, AssertionStr, llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), 0), newLPadBBBuilder.CreateGlobalStringPtr(F.getName().str())});
            
            //set unwind destination to new and edited landing pad block
            newLPadBBBuilder.CreateResume(newLP);
            Invoke->setUnwindDest(newLPadBB);
            
            //TODO: look into linkage again
          }
        }
      }
    }
  }
  return llvm::PreservedAnalyses::none();
}


}
}
