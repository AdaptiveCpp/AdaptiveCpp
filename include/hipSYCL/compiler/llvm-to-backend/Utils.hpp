/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#ifndef HIPSYCL_LLVM_TO_BACKEND_UTILS_HPP
#define HIPSYCL_LLVM_TO_BACKEND_UTILS_HPP

#include "hipSYCL/compiler/llvm-to-backend/LLVMToBackend.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Attributes.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/User.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Passes/PassBuilder.h>

namespace hipsycl {
namespace compiler {

template<class F>
class AtScopeExit {
public:
  AtScopeExit(F&& f)
  : Handler(f) {}

  ~AtScopeExit() {Handler();}
private:
  std::function<void()> Handler;
};

struct PassHandler {
  llvm::PassBuilder* PassBuilder;
  llvm::ModuleAnalysisManager* ModuleAnalysisManager;
};

inline llvm::Error loadModuleFromString(const std::string &LLVMIR, llvm::LLVMContext &ctx,
                                        std::unique_ptr<llvm::Module> &out) {

  auto buff = llvm::MemoryBuffer::getMemBuffer(LLVMIR, "", false);
  auto BC = llvm::parseBitcodeFile(*buff.get(), ctx);

  if(auto err = BC.takeError()) {
    return err;
  }

  out = std::move(BC.get());

  return llvm::Error::success();
}

template<class F>
inline void constructPassBuilder(F&& handler) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  handler(PB, LAM, FAM, CGAM, MAM);
}

template<class F>
inline void constructPassBuilderAndMAM(F&& handler) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  handler(PB, MAM);
}

// Attempts to determine the pointee type of a pointer function argument
// by investigating users, and looking for instructions that provide
// this information, such as getelementptr.
// This is particularly necessary due to LLVM's move to opaque pointers,
// where pointer types are no longer associated with the pointee type.
class FunctionArgPointeeTypeInferrer {
public:
  llvm::Type* tryInferType(llvm::Function* F, int ArgNo) {
    VisitedUsers.clear();

    if(llvm::Value* Arg = F->getArg(ArgNo)) {
      if(llvm::PointerType* PT = llvm::dyn_cast<llvm::PointerType>(Arg->getType())) {
        
        // If either byval or byref attributes are present, we can just look up
        // the pointee type directly.
        if(F->hasParamAttribute(ArgNo, llvm::Attribute::ByVal))
          return F->getParamAttribute(ArgNo, llvm::Attribute::ByVal).getValueAsType();
      
        if(F->hasParamAttribute(ArgNo, llvm::Attribute::ByRef))
          return F->getParamAttribute(ArgNo, llvm::Attribute::ByRef).getValueAsType();

        // Otherwise, we need to investigat    e uses of the argument to check
        // for clues regarding the pointee type.
        llvm::SmallSet<llvm::Value*, 1> UserPathsToInvestigate;
        UserPathsToInvestigate.insert(Arg);
        
        return this->extractTypeFromUserTree(UserPathsToInvestigate);

      } else {
        return Arg->getType();
      }
    }
    return nullptr;
  }

private:
  // Descend user tree in BFS fashion to find instructions that we can
  // determine pointee function argument types from.
  // It is assumed that all nodes \c Roots have already been investigated.
  // BFS order is important, because it guarantees that the more direct uses will
  // be preferred.
  template<unsigned int N>
  llvm::Type* extractTypeFromUserTree(const llvm::SmallSet<llvm::Value*, N>& Roots) {  
    if(Roots.empty())
      return nullptr;

    llvm::SmallSet<llvm::Value*, 16> NextUsers;

    for(llvm::Value* R : Roots) {
      for(llvm::Value* U : R->users()) {

        llvm::Type* CurrentResult = extractTypeFromUser(U, R, NextUsers);

        if(CurrentResult)
          return CurrentResult;
      }
    }

    return extractTypeFromUserTree(NextUsers);
  }

  template<class SmallSetT>
  llvm::Type* extractTypeFromUser(llvm::Value* U, llvm::Value* Parent, SmallSetT& UsersToInvestigate) {
    if(VisitedUsers.contains(U)) {
      return nullptr;
    }

    VisitedUsers.insert(U);

    if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
      return LI->getType();
    } else if (auto GEPI = llvm::dyn_cast<llvm::GetElementPtrInst>(U)) {
      return GEPI->getSourceElementType();
    } else if (auto EEI = llvm::dyn_cast<llvm::ExtractElementInst>(U)) {
      return EEI->getVectorOperand()->getType();
    } 
    else if (auto ACI = llvm::dyn_cast<llvm::AddrSpaceCastInst>(U)) {
      // Follow address space casts, we don't care about pointer address spaces
      for(llvm::User* NextLevelU : U->users()) {
        UsersToInvestigate.insert(NextLevelU);   
      }
    } else if(auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
      // Ugh, the value is forwarded as an argument into some other function, need
      // to continue looking there...
      int OperandNumber = 0;
      for (int i = 0; i < CI->getCalledFunction()->getFunctionType()->getNumParams(); ++i) {
        if(CI->getArgOperand(i) == Parent) {
          auto Arg = CI->getCalledFunction()->getArg(i);
          // Never, ever take into account the callee argument. This should never happen,
          // but if it does, it will go terribly because we will take into account users of functions,
          // not arguments anymore.
          if(!llvm::isa<llvm::Function>(Arg))
            UsersToInvestigate.insert(Arg);
        }
      }
    }

    return nullptr;
  }

  llvm::SmallSet<llvm::Value*, 32> VisitedUsers;
};

inline void forceByValOrByRefAttr(llvm::Module &M, llvm::Function *F, int ArgNo,
                                  llvm ::Attribute::AttrKind NewAttr,
                                  // If arg is a pointer and type is already known, provide it here -
                                  // otherwise we will try to infer the type if PointeeType is nullptr.
                                  llvm::Type* KnownPointeeType = nullptr) {

  assert(NewAttr == llvm::Attribute::AttrKind::ByRef ||
         NewAttr == llvm::Attribute::AttrKind::ByVal);

  llvm::Attribute::AttrKind ConflictingAttr = (NewAttr == llvm::Attribute::AttrKind::ByVal)
                                                  ? llvm::Attribute::AttrKind::ByRef
                                                  : llvm::Attribute::AttrKind::ByVal;

  if(ArgNo < F->getFunctionType()->getNumParams()) {
    llvm::PointerType *PT = llvm::dyn_cast<llvm::PointerType>(F->getArg(ArgNo)->getType());
    // Nothing to do for non-pointer types
    if(PT) {
      // Nothing to do if desired attr is already present
      if(F->hasParamAttribute(ArgNo, NewAttr)) {
        return;
      }

      llvm::Type* PointeeType = KnownPointeeType;
      if(!PointeeType) {
        FunctionArgPointeeTypeInferrer PTI;
        PointeeType = PTI.tryInferType(F, ArgNo);

        if(!PointeeType) {
          HIPSYCL_DEBUG_WARNING << "forceByValOrByRefAttr(): Pointee Type Inferrence failed, ByVal "
                                   "or ByRef attributes might be incorrect\n";
          return;
        }
      }

      if(F->hasParamAttribute(ArgNo, ConflictingAttr))
        F->removeParamAttr(ArgNo, ConflictingAttr);
      
      if(NewAttr == llvm::Attribute::AttrKind::ByVal) {
        F->addParamAttr(ArgNo, llvm::Attribute::getWithByValType(M.getContext(), PointeeType));
      } else {
        F->addParamAttr(ArgNo, llvm::Attribute::getWithByRefType(M.getContext(), PointeeType));
      }
    }
  }
}

inline void forceAllUsedPointerArgumentsToByVal(llvm::Module &M, llvm::Function *F) {
  for (int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if (llvm::isa<llvm::PointerType>(F->getArg(i)->getType())) {
      // If there are no uses, ignore - we may not be able to infer
      // the pointee type in that case to attach the ByVal attribute.
      if (F->getArg(i)->getNumUses() > 0) {
        forceByValOrByRefAttr(M, F, i, llvm::Attribute::ByVal);
      }
    }
  }
}

inline void forceAllUsedPointerArgumentsToByRef(llvm::Module &M, llvm::Function *F) {
  for (int i = 0; i < F->getFunctionType()->getNumParams(); ++i) {
    if (llvm::isa<llvm::PointerType>(F->getArg(i)->getType())) {
      // If there are no uses, ignore - we may not be able to infer
      // the pointee type in that case to attach the ByVal attribute.
      if (F->getArg(i)->getNumUses() > 0) {
        forceByValOrByRefAttr(M, F, i, llvm::Attribute::ByRef);
      }
    }
  }
}

}
}

#endif
