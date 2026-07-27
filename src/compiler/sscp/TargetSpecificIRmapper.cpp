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

#include "hipSYCL/compiler/sscp/TargetSpecificIRmapper.hpp"
#include "hipSYCL/common/debug.hpp"

#include "llvm/IR/IRBuilder.h"
#include <llvm/Demangle/Demangle.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <unordered_map>
#include <unordered_set>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace hipsycl::compiler {

// Replace __acpp_sscp_custom_intrinsic_<builtin_name> with llvm IR eqvivalent
llvm::PreservedAnalyses TargetSpecificIRMapper::run(llvm::Module &M,
                                                    llvm::ModuleAnalysisManager &MAM) {

  llvm::StringRef prefix = "__acpp_sscp_custom_intrinsic__";
  bool Changed = false;
  std::vector<llvm::Function *> FunctionsToRemove;
  for (llvm::Function &F : M) {
    if (F.hasName() && F.getName().contains(prefix)) {

      auto builtin_name = F.getName().drop_front(prefix.size()).str();
      std::replace(builtin_name.begin(), builtin_name.end(), '_', '.');
      llvm::Function *IntrinsicDecl = M.getFunction(builtin_name);

      // If not, create declaration
      if (!IntrinsicDecl) {
        IntrinsicDecl = llvm::Function::Create(
            F.getFunctionType(), llvm::GlobalValue::ExternalLinkage, builtin_name, &M);
      }
      std::vector<llvm::CallInst *> CallSites;
      for (llvm::User *U : F.users()) {
        if (auto *CI = llvm::dyn_cast<llvm::CallInst>(U)) {
          CallSites.push_back(CI);
        }
      }

      for (llvm::CallInst *CI : CallSites) {
        llvm::IRBuilder<> Builder(CI);

        // Collect arguments from the original call
        std::vector<llvm::Value *> Args;
        for (unsigned i = 0; i < CI->arg_size(); ++i) {
          Args.push_back(CI->getArgOperand(i));
        }

        // Emit the new intrinsic call forwarding the arguments
        llvm::CallInst *NewCall = Builder.CreateCall(IntrinsicDecl, Args);

        // Perfectly forward any debug info or metadata from the old call
        NewCall->copyMetadata(*CI);

        // Replace uses and clean up the old call instruction
        CI->replaceAllUsesWith(NewCall);
        CI->eraseFromParent();

        Changed = true;
      }

      // Mark the custom builtin declaration for removal now that all call
      // sites have been rewritten to use the actual intrinsic name.
      FunctionsToRemove.push_back(&F);
    }
  }

  for (llvm::Function *F : FunctionsToRemove) {
    F->eraseFromParent();
  }
  return Changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
}

} // namespace hipsycl::compiler
