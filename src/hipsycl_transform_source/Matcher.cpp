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
#include <vector>
#include <cassert>
#include "Matcher.hpp"
#include <iostream>
namespace hipsycl {
namespace transform {

CXXConstructCallerMatcher::CXXConstructCallerMatcher(const std::string& id)
  : _id{id}
{}

void
CXXConstructCallerMatcher::registerMatcher(clang::ast_matchers::MatchFinder& finder,
                                           CXXConstructCallerMatcher& handler,
                                           const std::string& id)
{
  handler = CXXConstructCallerMatcher{id};
  finder.addMatcher(clang::ast_matchers::decl().bind(id),
                    &handler);

}

void
CXXConstructCallerMatcher::run(
    const clang::ast_matchers::MatchFinder::MatchResult & result)
{
  if(const clang::Decl* function =
     result.Nodes.getNodeAs<clang::FunctionDecl>(_id))
  {
    std::vector<const clang::CXXConstructExpr*> constructExprs;

    const clang::Stmt* body = function->getBody();
    if(body)
    {
      findCXXConstructExprs(body, constructExprs);
    }

    for(auto expr : constructExprs)
    {
      const clang::Decl* constructor = expr->getConstructor();

      if(constructor)
      {
        bool isLambda = false;

        const auto* cdecl = clang::cast<clang::CXXConstructorDecl>(constructor);
        if(cdecl->getParent())
          isLambda = cdecl->getParent()->isLambda();

        if(!isLambda)
          _constructCallers[constructor].push_back(function);

        std::cout << "Constructor "
                  << constructor->getAsFunction()->getQualifiedNameAsString()
                  << "\n   called by: "
                  << function->getAsFunction()->getQualifiedNameAsString()
                  << std::endl;
      }
    }
  }
}

void
CXXConstructCallerMatcher::findCXXConstructExprs(const clang::Stmt* current,
                            std::vector<const clang::CXXConstructExpr*>& out)
{
  if(current)
  {
    for(auto child = current->child_begin();
        child != current->child_end(); ++child)
    {
      if(*child && clang::isa<clang::CXXConstructExpr>(*child))
        out.push_back(clang::cast<clang::CXXConstructExpr>(*child));

      findCXXConstructExprs(*child, out);
    }
  }
}

}
}
