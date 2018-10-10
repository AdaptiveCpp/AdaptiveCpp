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

#ifndef HIPSYCL_MATCHER_HPP
#define HIPSYCL_MATCHER_HPP

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"

#include <vector>
#include <unordered_map>
#include <memory>

namespace hipsycl {
namespace transform {

class CXXConstructCallerMatcher : public clang::ast_matchers::MatchFinder::MatchCallback
{
public:
  CXXConstructCallerMatcher() = default;
  CXXConstructCallerMatcher(const std::string& id);

  static void registerMatcher(clang::ast_matchers::MatchFinder& finder,
                              CXXConstructCallerMatcher& handler,
                              const std::string& id);

  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult & result) override;

  using ConstructCallerMapType =
    std::unordered_map<const clang::Decl*, std::vector<const clang::Decl*>>;

  const ConstructCallerMapType& getResults() const
  { return _constructCallers; }

private:
  void findCXXConstructExprs(const clang::Stmt* current,
                             std::vector<const clang::CXXConstructExpr*>& out);

  // Maps the constructor declaration to the declaration of
  // the calling function
  ConstructCallerMapType _constructCallers;
  std::string _id;
};

}
}

#endif
