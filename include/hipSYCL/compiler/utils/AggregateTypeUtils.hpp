/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_AGGREGATE_TYPE_UTILS_HPP
#define HIPSYCL_AGGREGATE_TYPE_UTILS_HPP

#include <llvm/IR/Type.h>
#include <llvm/ADT/SmallVector.h>
#include <vector>

namespace hipsycl {
namespace compiler {
namespace utils {


template <class F, class H>
void ForEachNonAggregateContainedTypeWithParentTypeMatcher(llvm::Type *T, F &&Handler,
                                      llvm::SmallVector<int, 16> CurrentIndices,
                                      llvm::SmallVector<llvm::Type*, 16> MatchedParentTypes,
                                      H&& ParentTypeMatcher) {
  if(!T)
    return;
  
  if(ParentTypeMatcher(T))
    MatchedParentTypes.push_back(T);
  
  if(T->isArrayTy()) {
    llvm::Type* ArrayElementT = T->getArrayElementType();
    for(int i = 0; i < T->getArrayNumElements(); ++i) {
      auto NextIndices = CurrentIndices;
      NextIndices.push_back(i);
      ForEachNonAggregateContainedTypeWithParentTypeMatcher(ArrayElementT, Handler, NextIndices,
                                                            MatchedParentTypes, ParentTypeMatcher);
    }
  } else if(T->isAggregateType()) {
    for(int i = 0; i < T->getNumContainedTypes(); ++i) {
      auto NextIndices = CurrentIndices;
      NextIndices.push_back(i);
      llvm::Type* SubType = T->getContainedType(i);

      ForEachNonAggregateContainedTypeWithParentTypeMatcher(SubType, Handler, NextIndices,
                                                            MatchedParentTypes, ParentTypeMatcher);
    }
  } else {
    Handler(T, CurrentIndices, MatchedParentTypes);
  }
}

template <class F>
void ForEachNonAggregateContainedType(llvm::Type *T, F &&Handler,
                                      llvm::SmallVector<int, 16> CurrentIndices = {}) {
  auto HandlerAdapter = [&](llvm::Type *T, auto Indices, auto MatchedParentTypes) {
    Handler(T, Indices);
  };
  ForEachNonAggregateContainedTypeWithParentTypeMatcher(T, HandlerAdapter, CurrentIndices, {},
                                                        [](auto *Type) { return false; });
}

}
}
}

#endif
