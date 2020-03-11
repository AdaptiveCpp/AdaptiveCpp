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
