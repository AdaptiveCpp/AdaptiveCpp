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

#ifndef ACPP_DYNAMIC_FUNCTION_SUPPORT_HPP
#define ACPP_DYNAMIC_FUNCTION_SUPPORT_HPP


#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>

#include <vector>

namespace hipsycl::compiler {

class DynamicFunctionIdentifactionPass
    : public llvm::PassInfoMixin<DynamicFunctionIdentifactionPass> {
public:
  
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

  const std::vector<std::string>& getDynamicFunctionNames() const {
    return DynamicFunctionNames;
  }

  const std::vector<std::string>& getDynamicFunctionDefinitionNames() const {
    return DynamicFunctionNames;
  }
private:
  std::vector<std::string> DynamicFunctionNames;
  std::vector<std::string> DynamicFunctionDefinitionNames;
};

class HostSideDynamicFunctionHandlerPass
    : public llvm::PassInfoMixin<HostSideDynamicFunctionHandlerPass> {
public:
  HostSideDynamicFunctionHandlerPass(const std::vector<std::string> &DynamicFunctionNames,
                                     const std::vector<std::string> &DynamicFunctionDefinitionNames)
      : DynamicFunctionNames{DynamicFunctionNames}, DynamicFunctionDefinitionNames{
                                                        DynamicFunctionDefinitionNames} {}


  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
private:
  const std::vector<std::string>& DynamicFunctionNames;
  const std::vector<std::string>& DynamicFunctionDefinitionNames;
};




}

#endif
