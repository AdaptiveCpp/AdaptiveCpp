/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay and contributors
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


#include "llvm/Analysis/CallGraph.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "CompilationState.hpp"

#include "hipSYCL/sycl/detail/debug.hpp"

#include <unordered_set>
#include <vector>

namespace hipsycl {
namespace compiler {

struct FunctionPruningIRPass : public llvm::ModulePass
{
  static char ID;

  FunctionPruningIRPass()
    : llvm::ModulePass(ID)
  {}

  llvm::StringRef getPassName() const
  {
    return "hipSYCL function pruning pass";
  }

  bool runOnModule(llvm::Module& M) override
  {
    if(!CompilationStateManager::getASTPassState().isDeviceCompilation())
      return false;

    for(auto& F : M.getFunctionList())
    {
      Functions.push_back(&F);
      // If this function has been marked as a kernel, add it to our list of
      // "entrypoints" that must remain present in the code.
      bool IsEntrypoint = CompilationStateManager::getASTPassState().isKernel(F.getName().str());
      // NOTE: We currently don't consider explicit device functions as entrypoints by default.
      // This is because they can cause problems in certain situations (needs
      // further investigation) and Clang currently does not support device object
      // linking anyways.
#if HIPSYCL_EXPERIMENTAL_DEVICE_LINKAGE
      IsEntrypoint = IsEntrypoint ||
        CompilationStateManager::getASTPassState().isExplicitlyDevice(F.getName().str());
#endif
      if (IsEntrypoint)
      {
        Entrypoints.push_back(&F);
      }
    }

    // Disable function pruning as it breaks standard math functions such as
    // fmin, which are distributed over vector types using a function pointer.
    // TODO: If no other issues come up, we should be able to remove function
    // pruning for good.
    // pruneUnusedFunctions(M);
    pruneUnusedGlobals(M);

    return true;
  }

private:
  void findChildrenOf(llvm::CallGraph& CG,
                      llvm::Function* F,
                      std::unordered_set<llvm::Function*>& Children)
  {
    if(F == nullptr)
      return;

    if(Children.find(F) != Children.end())
      return;

    Children.insert(F);
    auto Node = CG[F];
    for(auto C : *Node)
      findChildrenOf(CG, C.second->getFunction(), Children);
  }

  void pruneUnusedFunctions(llvm::Module& M)
  {
    llvm::CallGraph CG{M};

    // Find all functions used by entrypoints
    // Note that this does not include LLVM intrinsics
    std::unordered_set<llvm::Function*> FunctionsToKeep;
    for(llvm::Function* F : Entrypoints)
    {
      findChildrenOf(CG, F, FunctionsToKeep);
    }

    HIPSYCL_DEBUG_INFO << "IR Processing: Keeping " << FunctionsToKeep.size() << " out of "
                       << Functions.size() << " functions"<< std::endl;

    for(llvm::Function* F : FunctionsToKeep)
    {
      HIPSYCL_DEBUG_INFO << "IR Processing: Keeping function " << F->getName().str() << std::endl;
    }

    std::size_t NumRemovedFunctions = 0;
    // Remove all non-intrinsic functions that are not in FunctionsToKeep
    for(llvm::Function* F : Functions)
    {
      if (FunctionsToKeep.find(F) == FunctionsToKeep.end() && !F->isIntrinsic())
      {
        HIPSYCL_DEBUG_INFO << "IR Processing: Pruning unused function from device code: "
                           << F->getName().str() << std::endl;

        F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
        F->eraseFromParent();
        ++NumRemovedFunctions;
      }
    }

    HIPSYCL_DEBUG_INFO << "===> IR Processing: Function pruning complete, removed "
                      << NumRemovedFunctions << " function(s)."
                      << std::endl;
  }

  void pruneUnusedGlobals(llvm::Module& M)
  {

    HIPSYCL_DEBUG_INFO << " ****** Starting pruning of global variables ******"
                       << std::endl;

    std::vector<llvm::GlobalVariable*> VariablesForPruning;

    for(auto G =  M.global_begin(); G != M.global_end(); ++G)
    {
      llvm::GlobalVariable* GPtr = &(*G);
      if(canGlobalVariableBeRemoved(GPtr))
      {
        VariablesForPruning.push_back(GPtr);

        HIPSYCL_DEBUG_INFO << "IR Processing: Pruning unused global variable from device code: "
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

  bool canGlobalVariableBeRemoved(llvm::GlobalVariable* G) const
  {
    G->removeDeadConstantUsers();
    return G->getNumUses() == 0;
  }


  std::vector<llvm::Function*> Entrypoints;
  std::vector<llvm::Function*> Functions;
};

char FunctionPruningIRPass::ID = 0;

}
}

#endif
