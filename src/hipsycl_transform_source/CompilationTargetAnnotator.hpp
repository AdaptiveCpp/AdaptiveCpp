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
#include "clang/Analysis/CallGraph.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "Matcher.hpp"

namespace hipsycl {
namespace transform {

class CompilationTargetAnnotator
{
public:
  using CallerMapType =
      std::unordered_map<const clang::Decl*, std::vector<const clang::Decl*>>;

  CompilationTargetAnnotator(clang::Rewriter& rewriter,
                             clang::CallGraph& callGraph);


  void treatConstructsAsFunctionCalls(const CXXConstructCallerMatcher& constructCallers);

  void addAnnotations();

private:

  template<class AttributeType>
  bool containsAttributeForCompilation(const clang::Decl* f) const
  {
    auto functionAttributes = f->getAttrs();

    for(clang::Attr* currentAttrib : functionAttributes)
    {
      if(AttributeType::describedBy(currentAttrib))
        return true;

    }
    return false;
  }


  bool isHostFunction(const clang::Decl*) const;
  bool isDeviceFunction(const clang::Decl*) const;
  bool isKernelFunction(const clang::Decl*) const;

  bool canCallHostFunctions(const clang::Decl* f) const;
  bool canCallDeviceFunctions(const clang::Decl* f) const;

  void correctFunctionAnnotations(bool& host, bool& device, const clang::Decl* f);

  // These functions add the corresponding attribute to the attribute lists
  void markAsHost(const clang::Decl* f);
  void markAsDevice(const clang::Decl* f);

  void writeAnnotation(const clang::Decl* f, const std::string& annotation);

  clang::Rewriter& _rewriter;
  clang::CallGraph& _callGraph;

  CallerMapType _callers;

  std::unordered_map<const clang::Decl*, bool> _isFunctionProcessed;
  std::unordered_set<const clang::Decl*> _isFunctionCorrectedDevice;
  std::unordered_set<const clang::Decl*> _isFunctionCorrectedHost;
};

}
}

#endif
