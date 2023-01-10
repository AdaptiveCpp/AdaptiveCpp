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
#include <llvm/IR/Function.h>
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


class KernelFunctionParameterRewriter {
public:
  // Attribute that should be attached to by-value
  // pointer function arguments
  enum class ByValueArgAttribute {
    ByVal, ByRef
  };

  /// \param Attr Whether by-value arguments that are passed as pointers
  /// should have byval or byref attribute. Note: At least one of them
  /// must already be present.
  /// \param DesiredByValueArgAddressSpace Address space that pointers
  /// to by-value arguments should have
  /// \param DesiredPointerAddressSpace Address space that actual pointers
  /// should have
  KernelFunctionParameterRewriter(ByValueArgAttribute Attr, unsigned DesiredByValueArgAddressSpace,
                                  unsigned DesiredPointerAddressSpace)
      : Attr{Attr}, ByValueArgAddressSpace{DesiredByValueArgAddressSpace},
        PointerAddressSpace{DesiredPointerAddressSpace} {}

  void run(llvm::Module &M, const std::vector<std::string> &KernelNames,
           llvm::ModuleAnalysisManager &MAM) {
    for(auto KernelName : KernelNames) {
      if(auto* F = M.getFunction(KernelName)) {
        run(M, F);
      }
    }
    MAM.invalidate(M, llvm::PreservedAnalyses::none());
  }

  void run(llvm::Module& M, llvm::Function* F) {
    auto* OldFType = F->getFunctionType();
    std::string FunctionName = F->getName().str();
    F->setName(FunctionName+"_PreKernelParameterRewriting");
    // Make sure old function can be inlined
    auto OldLinkage = F->getLinkage();
    F->setLinkage(llvm::GlobalValue::InternalLinkage);

    llvm::SmallVector<llvm::Type*, 8> Params;
    for(int i = 0; i < OldFType->getNumParams(); ++i){
      llvm::Type* CurrentParamType = OldFType->getParamType(i);

      if(llvm::PointerType* PT = llvm::dyn_cast<llvm::PointerType>(CurrentParamType)) {
        bool HasByRefAttr = F->hasParamAttribute(i, llvm::Attribute::ByRef);
        bool HasByValAttr = F->hasParamAttribute(i, llvm::Attribute::ByVal);

        unsigned TargetAddressSpace = 0;
        if(!HasByRefAttr && !HasByValAttr) {
          // Regular pointer
          TargetAddressSpace = PointerAddressSpace;
        } else {
          // ByVal or ByRef - this probably means that
          // some struct is passed into the kernel by value.
          // (the attribute will be handled later)
          TargetAddressSpace = ByValueArgAddressSpace;
        }
        llvm::Type *NewT = llvm::PointerType::getWithSamePointeeType(PT, TargetAddressSpace);
        Params.push_back(NewT);
      
      } else {
        Params.push_back(CurrentParamType);
      }
    }


    llvm::FunctionType *FType = llvm::FunctionType::get(F->getReturnType(), Params, false);
    if (auto *NewF = llvm::dyn_cast<llvm::Function>(
            M.getOrInsertFunction(FunctionName, FType, F->getAttributes()).getCallee())) {

      // Fix ByVal and ByRef attributes (different backends require different
      // attributes for kernel arguments)
      for(int i = 0; i < NewF->getFunctionType()->getNumParams(); ++i) {
        bool HasByRefAttr = NewF->hasParamAttribute(i, llvm::Attribute::ByRef);
        bool HasByValAttr = NewF->hasParamAttribute(i, llvm::Attribute::ByVal);
        if(HasByRefAttr || HasByValAttr) {
          llvm::Attribute::AttrKind DesiredAttr = (Attr == ByValueArgAttribute::ByRef)
                                                     ? llvm::Attribute::ByRef
                                                     : llvm::Attribute::ByVal;
          llvm::Attribute::AttrKind PresentAttr =
              HasByRefAttr ? llvm::Attribute::ByRef : llvm::Attribute::ByVal;
          if(PresentAttr != DesiredAttr) {
            llvm::Type* ValT = F->getParamAttribute(i, PresentAttr).getValueAsType();
            assert(ValT);

            NewF->removeParamAttr(i, PresentAttr);
            if(DesiredAttr == llvm::Attribute::ByRef)
              NewF->addParamAttr(i, llvm::Attribute::getWithByRefType(M.getContext(), ValT));
            else
              NewF->addParamAttr(i, llvm::Attribute::getWithByValType(M.getContext(), ValT));            
          }
        }
      }

      // Now create function call to old function
      llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", NewF);
      llvm::SmallVector<llvm::Value*, 8> CallArgs;

      for(int i = 0; i < Params.size(); ++i) {
        if (llvm::PointerType *NewPT =
                llvm::dyn_cast<llvm::PointerType>(NewF->getArg(i)->getType())) {

          assert(llvm::isa<llvm::PointerType>(F->getArg(i)->getType()));
          llvm::PointerType *OldPT = llvm::dyn_cast<llvm::PointerType>(F->getArg(i)->getType());

          if (NewPT->getAddressSpace() != OldPT->getAddressSpace()) {
            auto *ASCastInst = new llvm::AddrSpaceCastInst{NewF->getArg(i), OldPT, "", BB};
            CallArgs.push_back(ASCastInst);
          } else {
            CallArgs.push_back(NewF->getArg(i));
          }
        } else {
          CallArgs.push_back(NewF->getArg(i));
        }
      }

      assert(CallArgs.size() == F->getFunctionType()->getNumParams());
      for(int i = 0; i < CallArgs.size(); ++i) {
        assert(CallArgs[i]->getType() == F->getFunctionType()->getParamType(i));
      }

      auto *Call = llvm::CallInst::Create(llvm::FunctionCallee(F), CallArgs,
                                            "", BB);
      llvm::ReturnInst::Create(M.getContext(), BB);
    }
  }
private:
  ByValueArgAttribute Attr;
  unsigned ByValueArgAddressSpace;
  unsigned PointerAddressSpace;
};


}
}

#endif
