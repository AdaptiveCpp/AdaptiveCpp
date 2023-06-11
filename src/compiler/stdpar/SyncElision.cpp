/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include "hipSYCL/compiler/stdpar/SyncElision.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>


namespace hipsycl {
namespace compiler {

llvm::PreservedAnalyses SyncElisionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static constexpr const char* ConsumeMarker = "__hipsycl_stdpar_consume_sync";
  static constexpr const char* OptimizableMarker = "__hipsycl_stdpar_optimizable_sync";

  llvm::SmallVector<llvm::CallInst*, 16> RemovableSyncCalls;

  for(auto& F : M) {
    for(auto& BB: F) {
      llvm::CallInst* PreviousOptimizableSyncInst = nullptr;
      for(auto& I: BB) {
        // We currently only support call instructions for consume and sync builtins, as
        // those are easier to handle (particularly regarding BB terminators)
        llvm::CallInst* CI = llvm::dyn_cast<llvm::CallInst>(&I);

        // If we have encountered any other instruction,
        // we may have to reset whether PreviousOptimizableSyncInst was set.
        bool NeedsReset = true;

        if(CI){
          if(auto* CalledF = CI->getCalledFunction()){
            if(CalledF->getName().compare(OptimizableMarker) == 0) {
              PreviousOptimizableSyncInst = CI;
              NeedsReset = false;
            } else if(CalledF->getName().compare(ConsumeMarker) == 0) {
              if(PreviousOptimizableSyncInst) {
                RemovableSyncCalls.push_back(PreviousOptimizableSyncInst);
                HIPSYCL_DEBUG_INFO << "[stdpar] SyncElision: Eliding synchronization in function "
                                   << F.getName().str() << "\n";
              }
            } else {
              // Ignore llvm.lifetime intrinsics
              if(CalledF->isIntrinsic() && CalledF->getName().startswith("llvm.lifetime"))
                NeedsReset = false;
            }
          }
        }

        if(NeedsReset)
          PreviousOptimizableSyncInst = nullptr;
      }
    }
  }

  auto* ConsumeFunction = M.getFunction(ConsumeMarker);
  if(ConsumeFunction) {
    llvm::SmallVector<llvm::CallInst*, 16> Calls;
    for(auto* U : ConsumeFunction->users()) {
      llvm::CallInst* CI = llvm::dyn_cast<llvm::CallInst>(U);
      if(CI) {
        Calls.push_back(CI);
      }
    }
    for(auto* CI : Calls)
      CI->eraseFromParent();
    ConsumeFunction->replaceAllUsesWith(llvm::UndefValue::get(ConsumeFunction->getType()));
    ConsumeFunction->eraseFromParent();
  }


  for(auto* Call : RemovableSyncCalls)
    Call->eraseFromParent();
  HIPSYCL_DEBUG_INFO << "[stdpar] SyncElision: Removed " << RemovableSyncCalls.size()
                     << " kernel wait() calls."
                     << "\n";

  return llvm::PreservedAnalyses::none();
}
}
}
