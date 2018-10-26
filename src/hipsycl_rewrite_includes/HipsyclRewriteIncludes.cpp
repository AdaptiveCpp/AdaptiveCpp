/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#include <string>
#include <cassert>
#include <cstdlib>

#include "CL/sycl/detail/debug.hpp"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "InclusionRewriter.hpp"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"

#include "../common/Paths.hpp"

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

class IncludeRewriter : public clang::PreprocessorFrontendAction
{
public:

  virtual std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance&,
                                                         StringRef) override
  {
    // This function should never be invoked by clang since we are only
    // working with the preprocessor
    assert(false);
    return nullptr;
  }

  virtual void ExecuteAction() override {}

  virtual void EndSourceFileAction() override
  {

    std::string sourceFile = this->getCurrentFile();
    std::string transformedFile = sourceFile.append(".inc.cpp");

    std::error_code error;
    llvm::raw_fd_ostream output{
      transformedFile,
      error,
      llvm::sys::fs::OpenFlags::F_None
    };


    hipsycl::transform::RewriteUserIncludesInInput(this->getCompilerInstance().getPreprocessor(),
                                                &output,
                                                this->getCompilerInstance().getPreprocessorOutputOpts());
  }

};

class HipsyclDiagConsumer : public clang::DiagnosticConsumer {
public:
  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                        const clang::Diagnostic& info) override {
    llvm::SmallVector<char, 128> message;
    info.FormatDiagnostic(message);
    std::string strMessage(message.begin(), message.end());
    std::string location = "<unknown location>";
    if(info.hasSourceManager())
      std::string location = info.getLocation().printToString(info.getSourceManager());

    if(level == clang::DiagnosticsEngine::Ignored ||
       level == clang::DiagnosticsEngine::Note ||
       level == clang::DiagnosticsEngine::Remark)
      HIPSYCL_DEBUG_INFO << location << ": " << strMessage << std::endl;
    else if(level == clang::DiagnosticsEngine::Warning)
      HIPSYCL_DEBUG_WARNING << location << ": " << strMessage << std::endl;
    else
    {
      llvm::errs() << "Fatal error: " << location << ": " << message << "\n";
      std::exit(-1);
    }
  }
};


int main(int argc, const char** argv)
{
  CommonOptionsParser op(argc, argv, llvm::cl::GeneralCategory);

  ClangTool tool(op.getCompilations(), op.getSourcePathList());

  ArgumentsAdjuster adjuster =
      [](const CommandLineArguments& args,StringRef) -> CommandLineArguments
  {
    CommandLineArguments modifiedArgs = args;

    modifiedArgs.push_back("-D__HIPSYCL_TRANSFORM__");

    std::string clangIncludeDir =
        hipsycl::paths::getClangIncludePath();
    if(!clangIncludeDir.empty())
      modifiedArgs.push_back("-I"+clangIncludeDir);

    return modifiedArgs;
  };
  tool.appendArgumentsAdjuster(adjuster);

  HipsyclDiagConsumer* diag = new HipsyclDiagConsumer;
  tool.setDiagnosticConsumer(diag);

  return tool.run(newFrontendActionFactory<IncludeRewriter>().get());

}
