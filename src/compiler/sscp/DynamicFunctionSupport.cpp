/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause



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
