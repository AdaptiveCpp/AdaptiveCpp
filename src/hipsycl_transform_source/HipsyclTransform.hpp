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

#ifndef HIPSYCL_TRANSFORM_HPP
#define HIPSYCL_TRANSFORM_HPP

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

#include "CompilationTargetAnnotator.hpp"
#include "Matcher.hpp"
#include "CallGraph.hpp"

namespace hipsycl {
namespace transform {

class CommandLineArgs
{
public:

  clang::tooling::CommandLineArguments
  consumeHipsyclArgs(const clang::tooling::CommandLineArguments& args);

  std::string getTransformDirectory() const;
  std::string getMainFilename() const;
private:
  std::string extractArg(const std::string& optionString) const;

  std::string _transformDirectory;
  std::string _mainFilename;
};

class Application
{
public:
  static CommandLineArgs& getCommandLineArgs()
  {
    return getInstance()._args;
  }

  static Application& getInstance()
  {
    static Application app;
    return app;
  }

private:
  Application(){}

  CommandLineArgs _args;
};


// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class HipsyclTransformASTConsumer : public clang::ASTConsumer {
public:
  HipsyclTransformASTConsumer(clang::Rewriter &R);

  virtual ~HipsyclTransformASTConsumer() override;

  // Override the method that gets called for each parsed top-level
  // declaration.

  bool HandleTopLevelDecl(clang::DeclGroupRef DR) override;

  void HandleTranslationUnit(clang::ASTContext &Ctx) override;

private:
  CallGraph _visitor;
  CXXConstructCallerMatcher _constructMatcher;

  clang::Rewriter& _rewriter;
};

class HipsyclTransfromFrontendAction : public clang::ASTFrontendAction {
public:
  HipsyclTransfromFrontendAction();

  void EndSourceFileAction() override;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI, clang::StringRef file) override;

private:
  clang::Rewriter _rewriter;
};




}
}

#endif
