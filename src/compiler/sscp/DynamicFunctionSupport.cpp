/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
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



#include "hipSYCL/compiler/sscp/DynamicFunctionSupport.hpp"
#include "hipSYCL/compiler/utils/ProcessFunctionAnnotationsPass.hpp"

#include <llvm/IR/PassManager.h>

namespace hipsycl::compiler {

llvm::PreservedAnalyses DynamicFunctionIdentifactionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {

  // First identify dynamic function infrastructure, which may need
  // separate handling on host and device.
  llvm::SmallVector<llvm::Function *> DynamicFunctions;
  llvm::SmallVector<llvm::Function *> DynamicFunctionDefinitions;

  const std::string DynamicFunctionAnnotation = "dynamic_function";
  const std::string DynamicFunctionDefinitionAnnotationArg0 = "dynamic_function_def_arg0";
  const std::string DynamicFunctionDefinitionAnnotationArg1 = "dynamic_function_def_arg1";

  utils::ProcessFunctionAnnotationPass PFA({DynamicFunctionAnnotation,
                                            DynamicFunctionDefinitionAnnotationArg0,
                                            DynamicFunctionDefinitionAnnotationArg1});
  PFA.run(M, AM);
  const auto &FoundAnnotations = PFA.getFoundAnnotations();
  auto DFIt = FoundAnnotations.find(DynamicFunctionAnnotation);
  auto DFD0It = FoundAnnotations.find(DynamicFunctionDefinitionAnnotationArg0);
  auto DFD1It = FoundAnnotations.find(DynamicFunctionDefinitionAnnotationArg1);

  auto RetrieveFunctionNames = [&](auto It, std::vector<std::string>& Output, int ArgNo){
    for (auto *F : It->second) {
      if (F) {
        for (auto *U : F->users()) {
          if (llvm::CallBase *CB = llvm::dyn_cast<llvm::CallBase>(U)) {
            if (CB->getNumOperands() >= 1) {
              if (auto *DF = llvm::dyn_cast<llvm::Function>(CB->getOperand(ArgNo))) {
                Output.push_back(DF->getName().str());
              } else {
                M.getContext().emitError(
                    CB, "Detected a dynamic_function or dynamic_function_definition construction "
                        "where the argument is not "
                        "directly a function; dynamic_function function pointer arguments do "
                        "not support indirection.");
              }
            }
          }
        }
      }
    }
  };

  if (DFIt != FoundAnnotations.end()) {
    RetrieveFunctionNames(DFIt, this->DynamicFunctionNames, 1);
  }

  if (DFD0It != FoundAnnotations.end()) {
    RetrieveFunctionNames(DFD0It, this->DynamicFunctionDefinitionNames, 1);
  }

  if (DFD1It != FoundAnnotations.end()) {
    RetrieveFunctionNames(DFD1It, this->DynamicFunctionDefinitionNames, 2);
  }

  return llvm::PreservedAnalyses::none();
}

llvm::PreservedAnalyses HostSideDynamicFunctionHandlerPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  // Provide dummy definitions to avoid linker issues
  for (const auto &FName : DynamicFunctionNames) {
    if (auto *F = M.getFunction(FName)) {
      if (F->isDeclaration()) {
        F->setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
        auto BB = llvm::BasicBlock::Create(M.getContext(), "entry", F);
        new llvm::UnreachableInst(M.getContext(), BB);
      }
    }
  }

  return llvm::PreservedAnalyses::none();
}


}
