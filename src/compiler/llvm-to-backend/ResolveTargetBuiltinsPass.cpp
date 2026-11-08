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
#include "hipSYCL/compiler/llvm-to-backend/ResolveTargetBuiltinsPass.hpp"
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

namespace hipsycl {
namespace compiler {

llvm::PreservedAnalyses ResolveTargetBuiltinsPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  bool Changed = false;
  for (auto &F : M) {
    if (llvmutils::starts_with(F.getName(), "__acpp___builtin_")) {
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
            if (llvm::Function *CalledF = Call->getCalledFunction()) {
              if (llvmutils::starts_with(CalledF->getName(), "llvm.")) {
                for (unsigned i = 0; i < Call->arg_size(); ++i) {
                  if (llvm::isa<llvm::Constant>(Call->getArgOperand(i)) && i < F.arg_size()) {
                    llvm::Argument *WrapperArg = F.getArg(i);
                    llvm::Value *Replacement = WrapperArg;
                    if (WrapperArg->getType() != Call->getArgOperand(i)->getType()) {
                       llvm::IRBuilder<> Builder(Call);
                       if (WrapperArg->getType()->isIntegerTy() && Call->getArgOperand(i)->getType()->isIntegerTy()) {
                         Replacement = Builder.CreateIntCast(WrapperArg, Call->getArgOperand(i)->getType(), false);
                       } else {
                         Replacement = Builder.CreateBitOrPointerCast(WrapperArg, Call->getArgOperand(i)->getType());
                       }
                    }
                    Call->setArgOperand(i, Replacement);
                    Changed = true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
