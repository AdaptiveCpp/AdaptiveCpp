/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
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
