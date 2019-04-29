/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_IR_HPP
#define HIPSYCL_IR_HPP


#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "CompilationState.hpp"

#include "CL/sycl/detail/debug.hpp"

namespace hipsycl {

struct FunctionPruningIRPass : public llvm::FunctionPass {
  static char ID;

  FunctionPruningIRPass() 
  : llvm::FunctionPass(ID) 
  {}

  virtual bool runOnFunction(llvm::Function &F) override
  {
    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
    {
      if(canFunctionBeRemoved(F))
      {
        HIPSYCL_DEBUG_INFO << "IR Processing: Stripping unneeded function from device code: " 
                          << F.getName().str() << std::endl;;
        FunctionsScheduledForRemoval.insert(&F);
      }
      else
        HIPSYCL_DEBUG_INFO << "IR Processing: Keeping function " << F.getName().str() << std::endl;
    }
    
    return false;
  }

  virtual bool doFinalization (llvm::Module& M) override
  {
    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
    {
      for(llvm::Function* F : FunctionsScheduledForRemoval)
      {
        F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
        F->eraseFromParent();
      }
      HIPSYCL_DEBUG_INFO << "===> IR Processing: Function pruning complete, removed " 
                        << FunctionsScheduledForRemoval.size() << " function(s)."
                        << std::endl;

      HIPSYCL_DEBUG_INFO << " ****** Starting pruning of global variables ******" 
                        << std::endl;

      std::vector<llvm::GlobalVariable*> VariablesForPruning;

      for(auto G =  M.global_begin(); G != M.global_end(); ++G)
      {
        

        llvm::GlobalVariable* GPtr = &(*G);
        if(canGlobalVariableBeRemoved(GPtr))
        {
          VariablesForPruning.push_back(GPtr);

          HIPSYCL_DEBUG_INFO << "IR Processing: Stripping unrequired global variable from device code: " 
                            << G->getName().str() << std::endl;
        }
      }

      for(auto G: VariablesForPruning)
      {
        G->replaceAllUsesWith(llvm::UndefValue::get(G->getType()));
        G->eraseFromParent();
      }
      HIPSYCL_DEBUG_INFO << "===> IR Processing: Pruning of globals complete, removed " 
                        << VariablesForPruning.size() << " global variable(s)."
                        << std::endl;
    }
    return true;
  }
private:
  bool canGlobalVariableBeRemoved(llvm::GlobalVariable* G) const
  {
    G->removeDeadConstantUsers();
    return G->getNumUses() == 0;
  }

  bool canUserBeRemoved(const llvm::User* U) const
  {
    if (const llvm::Instruction* I = llvm::dyn_cast<llvm::Instruction>(U)) {
      // If we're looking at an Instruction, look at the containing function instead
      // since we're interested if we're used by a kernel *function*
      const llvm::Function* F = I->getFunction();
      return canUserBeRemoved(F);
    }

    // A function can never be removed if it ended in device
    // code due to explicit attributes and not due to us
    // marking the function implicitly as __host__ __device__
    if (const llvm::Function *F = llvm::dyn_cast<llvm::Function>(U)) {
      if (!CompilationStateManager::getASTPassState().isImplicitlyHostDevice(
              F->getName().str())) {
        return false;
      }
    } 
    for(const llvm::User* NextLevelUser : U->users())
    {
      if(!canUserBeRemoved(NextLevelUser))
        return false;
    }

    return true;
  }

  bool canFunctionBeRemoved(const llvm::Function& F) const
  {
    return canUserBeRemoved(&F);
  }

  std::unordered_set<llvm::Function*> FunctionsScheduledForRemoval;
};

char FunctionPruningIRPass::ID = 0;

}

#endif
