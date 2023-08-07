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

#include <llvm/Support/Casting.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Instruction.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>


namespace hipsycl {
namespace compiler {

namespace {

bool accessesMemory(llvm::Instruction* I) {
  if (llvm::isa<llvm::StoreInst>(I) || llvm::isa<llvm::LoadInst>(I) ||
      llvm::isa<llvm::AtomicRMWInst>(I) || llvm::isa<llvm::AtomicCmpXchgInst>(I) ||
      llvm::isa<llvm::FenceInst>(I))
    return true;
  
  if(auto* CB = llvm::dyn_cast<llvm::CallBase>(I)) {
    if(auto* F = CB->getCalledFunction()) {
      if(F->isIntrinsic()) {
        // Currently we assume that all intrinsics apart from llvm.lifetime
        // may access memory. This is of course not true, but since intrinsics
        // are backend-specific it might be difficult to get a comprehensive list
        // of safe intrinsic
        if(!F->getName().startswith("llvm.lifetime"))
          return true;
      }
    }
  }

  return false;
}

bool isBranchingInst(llvm::Instruction* I) {
  if(auto* CB = llvm::dyn_cast<llvm::CallBase>(I))
    if(auto * F = CB->getCalledFunction())
      if(!F->isIntrinsic())
        return true;

  return llvm::isa<llvm::BranchInst>(I) || llvm::isa<llvm::CatchReturnInst>(I) ||
         llvm::isa<llvm::CatchSwitchInst>(I) || llvm::isa<llvm::CleanupReturnInst>(I) ||
         llvm::isa<llvm::IndirectBrInst>(I) || llvm::isa<llvm::ResumeInst>(I) ||
         llvm::isa<llvm::ReturnInst>(I) || llvm::isa<llvm::SwitchInst>(I);
}

bool instructionRequiresSync(llvm::Instruction* I) {
  assert(I);

  if(accessesMemory(I) || isBranchingInst(I))
    return true;
  
  return false;
}

constexpr const char* ConsumeMarker = "__hipsycl_stdpar_consume_sync";
constexpr const char* OptimizableMarker = "__hipsycl_stdpar_optimizable_sync";

}

llvm::PreservedAnalyses SyncElisionInliningPass::run(llvm::Module& M, llvm::ModuleAnalysisManager& AM) {

  auto InlineEachCaller = [&](llvm::Function *F) {
    if(!F)
      return;
    for (auto *U : F->users()) {
      if (auto *I = llvm::dyn_cast<llvm::CallBase>(U)) {
        if (auto *BB = I->getParent()) {
          if (auto *Caller = BB->getParent()) {
            if (Caller != F && !Caller->hasFnAttribute(llvm::Attribute::AlwaysInline)) {
              Caller->addFnAttr(llvm::Attribute::AlwaysInline);
            }
          }
        }
      }
    }
  };

  InlineEachCaller(M.getFunction(ConsumeMarker));
  InlineEachCaller(M.getFunction(OptimizableMarker));

  return llvm::PreservedAnalyses::all();
}

llvm::PreservedAnalyses SyncElisionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  if(auto* F = M.getFunction(ConsumeMarker)) {
    F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
  }
  if(auto* F = M.getFunction(OptimizableMarker)) {
    F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
  }

  llvm::SmallVector<llvm::CallInst*, 16> RemovableSyncCalls;
  std::size_t TotalNumSyncCalls = 0;

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
              ++TotalNumSyncCalls;
              PreviousOptimizableSyncInst = CI;
              NeedsReset = false;
            } else if(CalledF->getName().compare(ConsumeMarker) == 0) {
              if(PreviousOptimizableSyncInst) {
                RemovableSyncCalls.push_back(PreviousOptimizableSyncInst);
                HIPSYCL_DEBUG_INFO << "[stdpar] SyncElision: Eliding synchronization in function "
                                   << F.getName().str() << "\n";
              }
            } else {
              if(!instructionRequiresSync(&I))
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
                     << " kernel wait() calls out of " << TotalNumSyncCalls << " total calls"
                     << "\n";

  return llvm::PreservedAnalyses::none();
}
}
}
