/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay and contributors
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

#include "hipSYCL/compiler/sscp/HostKernelNameExtractionPass.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/common/debug.hpp"
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>

namespace hipsycl {
namespace compiler {

namespace {

static const char* SSCPExtractKernelNameIdentifier = "__hipsycl_sscp_extract_kernel_name";
}

llvm::PreservedAnalyses HostKernelNameExtractionPass::run(llvm::Module &M,
                                                          llvm::ModuleAnalysisManager &MAM) {

  llvm::SmallVector<llvm::Function*> SSCPKernelNameExtractionFunctions;

  for(auto& F : M) {
    if(F.getName().find(SSCPExtractKernelNameIdentifier) != std::string::npos) {
      SSCPKernelNameExtractionFunctions.push_back(&F);
      for(auto U : F.users()) {
        if(auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
          if(CI->arg_size() == 2) {
            // Arg 0 should be function pointer to the kernel entry point.
            std::string KernelName;
            if (llvm::Function *KernelFunc = llvm::dyn_cast<llvm::Function>(CI->getOperand(0))) {
              KernelName = KernelFunc->getName();
            } else {
              HIPSYCL_DEBUG_WARNING << "HostKernelNameExtractionPass: Could not find kernel name "
                                       "for __hipsycl_sscp_extract_kernel_name invocation: "
                                    << F.getName() << "\n";
            }
            
            // Arg 1 is the access to the global __hipsycl_sscp_kernel_name
            // variable. This might result in a ConstantExpr to do a getelementptr
            // Instruction.
            // We need to extract the global variable that is accessed, so that
            // we can access it.
            llvm::GlobalVariable* GV = nullptr;
            if(llvm::ConstantExpr* CE = llvm::dyn_cast<llvm::ConstantExpr>(CI->getArgOperand(1))) {
              for(auto O = CE->op_begin(); O != CE->op_end(); ++O) {
                if(llvm::GlobalVariable* G = llvm::dyn_cast<llvm::GlobalVariable>(O)) {
                  GV = G;
                }
              }
            } else if (llvm::GlobalVariable *G =
                           llvm::dyn_cast<llvm::GlobalVariable>(CI->getArgOperand(1))) {
              GV = G;
            }

            if(!GV) {
              HIPSYCL_DEBUG_WARNING
                  << "HostKernelNameExtractionPass: Could not find target global variable "
                     "for __hipsycl_sscp_extract_kernel_name invocation: "
                  << F.getName() << "\n";
            } else {
              
              IRConstant IRC{M, *GV};
              // If there are multiple calls (this can happen during e.g. stdpar malloc2usm callgraph
              // duplication, don't set again if we have already been set)
              if(!IRC.isInitialized()) {
                HIPSYCL_DEBUG_INFO << "HostKernelNameExtractionPass: Exposing kernel name "
                                  << KernelName << " in global symbol " << GV->getName() << "\n";
              
                // Now set GV to the kernel name
                IRC.set<std::string>(KernelName + '\0');
              }
            }
          
          } else {
            HIPSYCL_DEBUG_WARNING
                << "HostKernelNameExtractionPass: found __hipsycl_sscp_extract_kernel_name "
                   "invocation, but incorrect number of arguments ("
                << CI->arg_size() << ", should be 2)\n";
          }
        }
      }
    }    
  }

  for(llvm::Function* F : SSCPKernelNameExtractionFunctions) {
    // These functions are no longer needed
    llvm::SmallVector<llvm::CallBase*> KernelNameExtractionCalls;
    for(auto U : F->users()) {
      if(auto CI = llvm::dyn_cast<llvm::CallBase>(U)) {
        KernelNameExtractionCalls.push_back(CI);
      } else {
        HIPSYCL_DEBUG_WARNING
            << "HostKernelNameExtractionPass: found user of __hipsycl_sscp_extract_kernel_name() "
               "that is not a function call\n";
      }
    }

    for(auto Call : KernelNameExtractionCalls) {
      Call->replaceAllUsesWith(llvm::UndefValue::get(Call->getType()));
      Call->eraseFromParent();
    }

    F->replaceAllUsesWith(llvm::UndefValue::get(F->getType()));
    F->eraseFromParent();
  }

  return llvm::PreservedAnalyses::none();
}

}
}
