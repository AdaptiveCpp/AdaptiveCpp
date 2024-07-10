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
#ifndef HIPSYCL_FRONTEND_PLUGIN_HPP
#define HIPSYCL_FRONTEND_PLUGIN_HPP

#include "Frontend.hpp"

#include "clang/AST/AST.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"

namespace hipsycl {
namespace compiler {

class FrontendASTAction : public clang::PluginASTAction {
  
protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI,
                                                        llvm::StringRef) override 
  {
    return std::make_unique<FrontendASTConsumer>(CI);
  }

  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override 
  {
    return true;
  }

  bool BeginInvocation (clang::CompilerInstance &CI) override
  {
    // Unfortunately BeginInvocation does not seem to be called :(
    CI.getInvocation().getPreprocessorOpts().addMacroDef(
      "__sycl_kernel=__attribute__((diagnose_if(false,\"hipsycl_kernel\",\"warning\")))");
    CI.getInvocation().getPreprocessorOpts().addMacroDef("HIPSYCL_CLANG=1");

    return true;
  }

  void PrintHelp(llvm::raw_ostream& ros) {}

  clang::PluginASTAction::ActionType getActionType() override 
  {
    return AddBeforeMainAction;
  }

};

}
}

#endif
