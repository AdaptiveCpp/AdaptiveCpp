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


#include "hipSYCL/compiler/stdpar/MallocToUSM.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"



#include <llvm/IR/Instructions.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Analysis/CallGraph.h>


namespace hipsycl {
namespace compiler {

namespace {

bool NameStartsWithItaniumIdentifier(llvm::StringRef Name, llvm::StringRef Identifier) {
  auto FirstNumber = Name.find_first_of("0123456789");
  auto IdentifierPos = Name.find(std::to_string(Identifier.size())+Identifier.str());

  if (FirstNumber == std::string::npos || IdentifierPos == std::string::npos)
    return false;
  
  return FirstNumber == IdentifierPos;
}

bool NameContainsItaniumIdentifier(llvm::StringRef Name, llvm::StringRef Identifier) {
  return Name.contains(std::to_string(Identifier.size())+Identifier.str());
}

bool isForbiddenCaller(llvm::Function* F) {
  llvm::StringRef Name = F->getName();
  if(!Name.startswith("_Z"))
    return false;
  
  //if(!NameContainsItaniumIdentifier(Name, "hipsycl"))
  //  return false;
  
  if(NameStartsWithItaniumIdentifier(Name, "hipsycl"))
    return true;
  
  // This is a shared_ptr with a mention of hipsycl::
  // so most likely carrying a hipsycl datatype.
  /*if(NameStartsWithItaniumIdentifier(Name, "shared_ptr"))
    return true;
  if(NameStartsWithItaniumIdentifier(Name, "weak_ptr"))
    return true;*/
  
  return false;
}


template <class Handler>
void forAllForbiddenCallers(
    const llvm::DenseMap<llvm::Function *, llvm::SmallPtrSet<llvm::Function *, 16>> &CallerMap,
    llvm::SmallPtrSet<llvm::Function *, 16> &VisitedFunctions, llvm::Function *Current,
    Handler &&H) {

  //std::cout << "Investigating " << Current->getName().str() << "\n";
  if(VisitedFunctions.contains(Current))
    return;
  VisitedFunctions.insert(Current);

  auto It = CallerMap.find(Current);
  if(It == CallerMap.end()) {
    //std::cout << "(No callers)" << std::endl;
    return;
  }

  for(auto* Caller : It->getSecond()) { 
    if(isForbiddenCaller(Caller)) {
      H(Caller, Current);
    } else {
      forAllForbiddenCallers(CallerMap, VisitedFunctions, Caller, H);
    }
  }
}
}

llvm::PreservedAnalyses MallocToUSMPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  static constexpr const char* MemoryManagementIdentifier = "hipsycl_stdpar_memory_management";
  static constexpr const char* VisibilityIdentifier = "hipsycl_stdpar_mmgmt_visibility";
  llvm::SmallVector<llvm::Function*, 16> ManagedMemoryManagementFunctions;
  llvm::SmallVector<llvm::Function*, 16> MemoryManagementVisibilityFunctions;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function* F, llvm::StringRef Annotation){
    if(F) {
      if(Annotation.compare(MemoryManagementIdentifier) == 0) {
        HIPSYCL_DEBUG_INFO
            << "[stdpar] MallocToUSM: Found new memory management function definition: "
            << F->getName() << "\n";
        ManagedMemoryManagementFunctions.push_back(F);
        
      }
      if(Annotation.compare(VisibilityIdentifier) == 0) {
        MemoryManagementVisibilityFunctions.push_back(F);
      }
    }
  });


  llvm::CallGraph CG{M};

  for(auto& F: M) {
    if(isForbiddenCaller(&F) && F.getLinkage() != llvm::GlobalValue::ExternalLinkage) {
      F.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }

  for(auto& F: M)
    CG.addToCallGraph(&F);

  llvm::DenseMap<llvm::Function*, llvm::SmallPtrSet<llvm::Function*, 16>> FunctionCallers;

  for(auto& F: M) {
    llvm::CallGraphNode* CGN = CG.getOrInsertFunction(&F);
    if(CGN) {
      for(unsigned i = 0; i < CGN->size(); ++i){
        auto* CalledFunction = (*CGN)[i]->getFunction();
        if(CalledFunction) {
          FunctionCallers[CalledFunction].insert(&F);
        }
      }
    }
  }

  llvm::DenseMap<llvm::Function *, llvm::SmallPtrSet<llvm::Function *, 16>>
      FunctionsContainingCallsNeedingGuards;

  auto FindForbiddenCallers = [&](llvm::Function* F) {
    llvm::SmallPtrSet<llvm::Function *, 16> VisitedFunctions;
    forAllForbiddenCallers(FunctionCallers, VisitedFunctions, F,
                           [&](llvm::Function *Caller, llvm::Function *Callee) {
                             FunctionsContainingCallsNeedingGuards[Caller].insert(Callee);
                           });
  };
  // We need to handle the case
  // a) When we have a call from a forbidden function to one
  // of the memory management functions
  for(auto* F : ManagedMemoryManagementFunctions) {
    FindForbiddenCallers(F);
  }
  // b) When we have a call from a forbidden function to
  // an undefined function, as that function might do
  // memory management in its implementation
  for(auto& F : M) {
    if(F.isDeclaration() && !F.isIntrinsic()) {
      FindForbiddenCallers(&F);
    }
  }

  llvm::Function* StartGuard = M.getFunction("__hipsycl_stdpar_push_disable_usm");
  llvm::Function* EndGuard = M.getFunction("__hipsycl_stdpar_pop_disable_usm");

  if(!StartGuard || !EndGuard) {
    HIPSYCL_DEBUG_WARNING << "[stdpar] MallocToUSM: Could not find malloc guard functions in "
                             "module. Memory allocations might not work as expected.\n";
    return llvm::PreservedAnalyses::none();
  }

  StartGuard->setLinkage(llvm::GlobalValue::InternalLinkage);
  EndGuard->setLinkage(llvm::GlobalValue::InternalLinkage);

  // Now find all calls that need guarding
  llvm::SmallVector<llvm::CallBase*> CallsNeedingGuards;
  for(auto Entry: FunctionsContainingCallsNeedingGuards) {
    llvm::Function* F = Entry.getFirst();

    for(auto& BB : *F) {
      for(auto& I : BB) {
        // Look for all types of calls
        if(auto* CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
          // If the call is to one of the functions that need guarding
          if(CB->getCalledFunction() && Entry.getSecond().contains(CB->getCalledFunction())) {
            HIPSYCL_DEBUG_INFO
                << "[stdpar] MallocToUSM: Will plant USM allocation exclusion guard into "
                << F->getName().str() << " for call to " << CB->getCalledFunction()->getName().str()
                << "\n";
            CallsNeedingGuards.push_back(CB);
          }
        }
      }  
    }

  }

  for(auto* CB : CallsNeedingGuards) {
    llvm::SmallVector<llvm::Value*> EmptyArgList;
    // We need to insert the start call before the call instruction, and the
    // end call afterwards.
    // The start call is easy:
    llvm::CallInst::Create(llvm::FunctionCallee(StartGuard), EmptyArgList, "", CB);
    // For the end call, it depends on the exact type of call instruction we have:
    if(auto* CI = llvm::dyn_cast<llvm::CallInst>(CB)) {
      auto NextInst = CI->getNextNonDebugInstruction();
      if(!NextInst) {
        HIPSYCL_DEBUG_WARNING << "[stdpar] MallocToUSM: Attempted to exclude call to "
                              << (CI->getCalledFunction() ? CI->getCalledFunction()->getName().str()
                                                          : "(nullptr)")
                              << " from USM memory management, but the call does not have "
                                 "succeeding instructions to terminate exclusion.\n";
        NextInst = CI;
      }
      llvm::CallInst::Create(llvm::FunctionCallee(EndGuard), EmptyArgList, "", NextInst);
    } else if(auto* II = llvm::dyn_cast<llvm::InvokeInst>(CB)) {
      if(II->getNormalDest()) {
        llvm::CallInst::Create(llvm::FunctionCallee(EndGuard), EmptyArgList, "",
                               II->getNormalDest()->getFirstNonPHI());
      }
      if(II->getLandingPadInst()) {
        // We need to ensure that the landing pad instruction remains the
        // first in the block, so insert after that.
        llvm::CallInst::Create(llvm::FunctionCallee(EndGuard), EmptyArgList, "",
                               II->getLandingPadInst()->getNextNonDebugInstruction());
      }
    } else {
      HIPSYCL_DEBUG_WARNING << "[stdpar] MallocToUSM: Encountered unhandled call type; cannot "
                               "insert memory allocation exclusion guards.\n";
      // Better make sure to at least undo the start guard, so that we at least have a consistent state..
      llvm::CallInst::Create(llvm::FunctionCallee(EndGuard), EmptyArgList, "", CB);
    }
  }

  // Internalize memory management definitions
  for(auto* F: MemoryManagementVisibilityFunctions) {
    //F->setVisibility(llvm::GlobalValue::HiddenVisibility);
    F->setLinkage(llvm::GlobalValue::InternalLinkage);
  }

  return llvm::PreservedAnalyses::none();
}
}
}
