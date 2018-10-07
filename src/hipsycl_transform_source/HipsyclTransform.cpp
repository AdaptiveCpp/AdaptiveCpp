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


namespace hipsycl {
namespace transform {

HipsyclTransformASTConsumer::HipsyclTransformASTConsumer(clang::Rewriter& R)
  : _visitor{R}
{}

HipsyclTransformASTConsumer::~HipsyclTransformASTConsumer()
{}

bool
HipsyclTransformASTConsumer::HandleTopLevelDecl(clang::DeclGroupRef DR)
{
  for (clang::DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {

    _visitor.TraverseDecl(*b);

    //(*b)->dump();
  }
  return true;
}

void HipsyclTransformASTConsumer::HandleTranslationUnit(clang::ASTContext&)
{
  _visitor.addAnnotations();
}

HipsyclTransfromFrontendAction::HipsyclTransfromFrontendAction()
{}

void
HipsyclTransfromFrontendAction::EndSourceFileAction()
{
  clang::SourceManager &sourceMgr = _rewriter.getSourceMgr();

  std::string source_file = this->getCurrentFile();
  std::string transformed_file = source_file.append(".transformed.cpp");

  std::error_code error;
  llvm::raw_fd_ostream output{
    transformed_file,
    error,
    llvm::sys::fs::OpenFlags::F_None
  };

  // Now emit the rewritten buffer.
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
