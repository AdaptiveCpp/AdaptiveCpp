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
#ifndef ACPP_DEVICE_ASSERT_PASS_HPP
#define ACPP_DEVICE_ASSERT_PASS_HPP

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace hipsycl {
namespace compiler {

class DeviceAssertPass : public llvm::PassInfoMixin<DeviceAssertPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  //static members relevant for compiler passes touching device assertions (e.g. DeviceAssertPass.cpp, ExceptionToAssertion.cpp)

  // void __assert_fail(const char * assertion, const char * file, unsigned int line, const char * function);
  // void  __glibcxx_assert_fail(const char* __file, int __line, const char* __function, const char* __condition)
  //_ZSt21__glibcxx_assert_failPKciS0_S0_(ptr noundef, i32 noundef, ptr noundef, ptr noundef) local_unnamed_addr #0
  static llvm::Type* getCharPtrType(llvm::Module* M, unsigned AS = 0) {
  #if LLVM_VERSION_MAJOR >= 16
    return llvm::PointerType::get(M->getContext(), AS);
  #else
    return llvm::PointerType::get(llvm::Type::getInt8Ty(M->getContext()), AS);
  #endif
  }
   
  static constexpr const char* AssertFailBuiltinName = "__acpp_sscp_assert_fail";
  static constexpr const char* GlibcxxAssertFailBuiltinName = "__acpp_sscp_glibcxx_assert_fail";
  static llvm::Function* getAssertFailBuiltin(llvm::Module& M) {
    if(auto* F = M.getFunction(AssertFailBuiltinName))
      return F;
    else {
      llvm::SmallVector<llvm::Type*> ParamTs;
      // assertion
      ParamTs.push_back(getCharPtrType(&M));
      // file
      ParamTs.push_back(getCharPtrType(&M));
      // line
      ParamTs.push_back(llvm::Type::getInt32Ty(M.getContext()));
      // function name
      ParamTs.push_back(getCharPtrType(&M));

      auto FC = M.getOrInsertFunction(AssertFailBuiltinName,
                                      llvm::FunctionType::get(llvm::Type::getVoidTy(M.getContext()),
                                                              llvm::ArrayRef<llvm::Type *>{ParamTs},
                                                              false));
      llvm::Function *NewDeclaration = llvm::dyn_cast<llvm::Function>(FC.getCallee());
      NewDeclaration->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      return NewDeclaration;
    }
  }
  static llvm::Function* getGlibcxxAssertFailBuiltin(llvm::Module& M) {
      if(auto* F = M.getFunction(GlibcxxAssertFailBuiltinName))
        return F;
    else {
      llvm::SmallVector<llvm::Type*> ParamTs;
      // file
      ParamTs.push_back(getCharPtrType(&M));
      // line
      ParamTs.push_back(llvm::Type::getInt32Ty(M.getContext()));
      // function name
      ParamTs.push_back(getCharPtrType(&M));
      // assertion
      ParamTs.push_back(getCharPtrType(&M));

      auto FC = M.getOrInsertFunction(GlibcxxAssertFailBuiltinName,
                                      llvm::FunctionType::get(llvm::Type::getVoidTy(M.getContext()),
                                                              llvm::ArrayRef<llvm::Type *>{ParamTs},
                                                              false));
      llvm::Function *NewDeclaration = llvm::dyn_cast<llvm::Function>(FC.getCallee());
      NewDeclaration->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
      return NewDeclaration;
    }
  }
};

}
}


#endif

