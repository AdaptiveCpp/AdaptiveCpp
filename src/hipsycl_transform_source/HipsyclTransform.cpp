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


#include "HipsyclTransform.hpp"

#include "clang/Rewrite/Frontend/Rewriters.h"

#include <cstdlib>
#include <stdexcept>

namespace hipsycl {
namespace transform {


std::string
CommandLineArgs::extractArg(const std::string& optionString) const
{
  std::size_t pos;
  if((pos = optionString.find("=")) != std::string::npos)
  {
    return optionString.substr(pos+1);
  }
  else
    throw std::invalid_argument("Option "+optionString+" is missing an argument.");
}

clang::tooling::CommandLineArguments
CommandLineArgs::consumeHipsyclArgs(
    const clang::tooling::CommandLineArguments& args)
{
  clang::tooling::CommandLineArguments modifiedArgs;

  _transformDirectory = "";
  _mainFilename = "";

  for(std::size_t i = 0; i < args.size(); ++i)
  {
    if(args[i].find("--hipsycl-transform-dir") == 0)
    {
      _transformDirectory = extractArg(args[i]);
    }
    else if(args[i].find("--hipsycl-main-output-file") == 0)
    {
      _mainFilename = extractArg(args[i]);
    }
    else
      modifiedArgs.push_back(args[i]);

    if(_mainFilename.empty())
      _mainFilename = "hipsycl_transformed_main.cpp";
  }

  return modifiedArgs;
}

std::string
CommandLineArgs::getTransformDirectory() const
{ return _transformDirectory; }

std::string
CommandLineArgs::getMainFilename() const
{ return _mainFilename; }

std::string
Application::getAbsoluteMainFilename() const
{
  std::string fullFilename = this->_args.getTransformDirectory();

  if(!fullFilename.empty())
  {
    if(fullFilename[fullFilename.size()-1] != '/')
    {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
      if(fullFilename[fullFilename.size()-1] != '\\')
#endif
        fullFilename += '/';
    }
  }
  fullFilename += this->_args.getMainFilename();

  return fullFilename;
}


HipsyclTransformASTConsumer::HipsyclTransformASTConsumer(clang::Rewriter& R)
  : _rewriter{R}
{
}

HipsyclTransformASTConsumer::~HipsyclTransformASTConsumer()
{}

bool
HipsyclTransformASTConsumer::HandleTopLevelDecl(clang::DeclGroupRef DR)
{
  return true;
}

void HipsyclTransformASTConsumer::HandleTranslationUnit(clang::ASTContext& ctx)
{
  _visitor.TraverseDecl(ctx.getTranslationUnitDecl());

  clang::ast_matchers::MatchFinder finder;
  CXXConstructCallerMatcher::registerMatcher(finder,
                                             _constructMatcher,
                                             "construct_matcher");


  finder.matchAST(ctx);

  CompilationTargetAnnotator annotationCorrector{_rewriter, _visitor};
  annotationCorrector.treatConstructsAsFunctionCalls(_constructMatcher);
  annotationCorrector.addAnnotations();
}

HipsyclTransfromFrontendAction::HipsyclTransfromFrontendAction()
{}

void
HipsyclTransfromFrontendAction::EndSourceFileAction()
{
  clang::SourceManager &sourceMgr = _rewriter.getSourceMgr();

  std::string transformedFile =
      Application::getInstance().getAbsoluteMainFilename();

  std::error_code error;
  llvm::raw_fd_ostream output{
    transformedFile,
    error,
    llvm::sys::fs::OpenFlags::F_None
  };

  _rewriter.getEditBuffer(sourceMgr.getMainFileID()).write(output);
}

std::unique_ptr<clang::ASTConsumer>
HipsyclTransfromFrontendAction::CreateASTConsumer(clang::CompilerInstance &CI,
                                                  clang::StringRef)
{
  _rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
  return llvm::make_unique<HipsyclTransformASTConsumer>(_rewriter);
}

}
}
