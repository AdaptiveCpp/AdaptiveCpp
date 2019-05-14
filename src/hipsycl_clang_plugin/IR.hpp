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

#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace hipsycl {

class CallGraph
{
public:
  void addNodeIfNotPresent(llvm::Function* F)
  {
    if(Callees.find(F) == Callees.end())
    {
      Callees[F] = std::vector<llvm::Function *>{};
    }
  }

  void addCallee(llvm::Function* Node, llvm::Function* Callee)
  {
    Callees[Node].push_back(Callee);
  }

  const std::vector<llvm::Function*>& getCallees(llvm::Function* Node) const
  {
    return Callees.at(Node);
  }

  /// Finds all nodes in the subgraph reachable from F and stores
  /// the result in \c Children.
  /// If \c Children is non-empty, its contents will be interpreted
  /// as functions of which we are already certain that they are children.
  ///
  /// This behaviour can be used to use several calls to this function
  /// to obtain a set of all children of several functions.
  void findChildrenOf(llvm::Function* F, 
                      std::unordered_set<llvm::Function*>& Children)
  {
    if(Callees.find(F) == Callees.end())
      return;
    
    this->findChildrenOfImpl(F, Children);
  }                                

private:
  void findChildrenOfImpl(llvm::Function* Current,
                    std::unordered_set<llvm::Function*>& Children)
  {
    // If this function has already been found by a previous invocation,
    // abort to avoit getting stuck in cycles of the call graph.
    if(Children.find(Current) != Children.end())
      return;
    
    Children.insert(Current);

    for(auto Child : Callees[Current])
      findChildrenOfImpl(Child, Children);
  }

  std::unordered_map<llvm::Function*, std::vector<llvm::Function*>> Callees;
};

struct FunctionPruningIRPass : public llvm::FunctionPass {
  static char ID;

  FunctionPruningIRPass() 
  : llvm::FunctionPass(ID) 
  {}

  virtual bool runOnFunction(llvm::Function &F) override
  {
    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
    {
      Functions.push_back(&F);
      // Make sure that all functions are represented in the call graph,
      // even if they are not actually used.
      CG.addNodeIfNotPresent(&F);

      for(llvm::User* U : F.users())
      {
        if (llvm::Function* Caller = llvm::dyn_cast<llvm::Function>(U))
          CG.addCallee(Caller, &F);
        else if (llvm::Instruction* I = llvm::dyn_cast<llvm::Instruction>(U)) {
          // If we're looking at an Instruction, look at the containing function instead
          // since we're interested if we're used by a kernel *function*
          CG.addCallee(I->getFunction(), &F);
        }
      }
      // If this function has been marked as a kernel or explicit device function,
      // add it to our list of "entrypoints" that must remain present in the code
      if (CompilationStateManager::getASTPassState().isKernel(F.getName().str()) 
        || CompilationStateManager::getASTPassState().isExplicitlyDevice(F.getName().str()))
      {
        Entrypoints.push_back(&F);
      }
    }
  
    return false;
  }

  virtual bool doFinalization (llvm::Module& M) override
  {
    if(CompilationStateManager::getASTPassState().isDeviceCompilation())
    {
      // Find all functions depending on entrypoints (kernels or functions with
      // explicit __device__ attributes)
      std::unordered_set<llvm::Function*> FunctionsToKeep;
      for(llvm::Function* F : Entrypoints)
        CG.findChildrenOf(F, FunctionsToKeep);

      for(llvm::Function* F: FunctionsToKeep){
        HIPSYCL_DEBUG_INFO << "IR Processing: Keeping function " << F->getName().str() << std::endl;
      }

      std::size_t NumRemovedFunctions = 0;
      // Remove all functions that are not in FunctionsToKeep
      for(llvm::Function* F : Functions)
      {
        if (FunctionsToKeep.find(F) == FunctionsToKeep.end())
        {
          HIPSYCL_DEBUG_INFO
              << "IR Processing: Stripping unneeded function from device code: "
              << F->getName().str() << std::endl;
          
          F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
          F->eraseFromParent();
          ++NumRemovedFunctions;
        }
      }
      
      HIPSYCL_DEBUG_INFO << "===> IR Processing: Function pruning complete, removed " 
                        << NumRemovedFunctions << " function(s)."
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


  std::vector<llvm::Function*> Entrypoints;
  std::vector<llvm::Function*> Functions;
  CallGraph CG;
};

char FunctionPruningIRPass::ID = 0;

}

#endif
