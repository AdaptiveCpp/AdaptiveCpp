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

#ifndef HIPSYCL_COMPILATION_TARGET_ANNOTATOR_HPP
#define HIPSYCL_COMPILATION_TARGET_ANNOTATOR_HPP

#include <string>
#include <unordered_map>
#include <unordered_set>
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Rewrite/Core/Rewriter.h"

namespace hipsycl {
namespace transform {

class CompilationTargetAnnotatingASTVisitor
    : public clang::RecursiveASTVisitor<CompilationTargetAnnotatingASTVisitor>
{
public:
  using CallerMapType =
      std::unordered_map<clang::FunctionDecl*, std::vector<clang::FunctionDecl*>>;

  CompilationTargetAnnotatingASTVisitor(clang::Rewriter& rewriter);

  bool VisitCallExpr(clang::CallExpr*);
  bool VisitFunctionDecl(clang::FunctionDecl*);

  void addAnnotations();

private:
  bool containsTargetAttribute(clang::FunctionDecl* f, const std::string& target) const;
  bool isHostFunction(clang::FunctionDecl*) const;
  bool isDeviceFunction(clang::FunctionDecl*) const;
  bool isKernelFunction(clang::FunctionDecl*) const;

  bool canCallHostFunctions(clang::FunctionDecl* f) const;
  bool canCallDeviceFunctions(clang::FunctionDecl* f) const;

  void correctFunctionAnnotations(bool& host, bool& device, clang::FunctionDecl* f);

  // These functions add the corresponding attribute to the attribute lists
  void markAsHost(clang::FunctionDecl* f);
  void markAsDevice(clang::FunctionDecl* f);

  void writeAnnotation(clang::FunctionDecl* f, const std::string& annotation);

  clang::Rewriter& _rewriter;
  clang::FunctionDecl* _currentFunction;
  CallerMapType _callers;

  std::unordered_map<clang::FunctionDecl*, bool> _isFunctionProcessed;
  std::unordered_set<clang::FunctionDecl*> _isFunctionCorrectedDevice;
  std::unordered_set<clang::FunctionDecl*> _isFunctionCorrectedHost;
};

}
}

#endif
