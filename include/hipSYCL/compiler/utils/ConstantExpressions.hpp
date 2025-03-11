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


#ifndef ACPP_COMPILER_UTILS_CONSTEXPR_HPP
#define ACPP_COMPILER_UTILS_CONSTEXPR_HPP

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Constants.h>

namespace hipsycl { 
namespace compiler {
namespace utils {

namespace constexpr_unfolding {

inline void collectFunctionUsers(llvm::ConstantExpr *CE,
                                 llvm::SmallPtrSet<llvm::Function *, 16> &Functions) {
  for (auto *User : CE->users()) {
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(User)) {
      auto *BB = I->getParent();
      if (BB && BB->getParent()) {
        Functions.insert(BB->getParent());
      }
    } else if(auto* NestedCE = llvm::dyn_cast<llvm::ConstantExpr>(User)) {
      collectFunctionUsers(NestedCE, Functions);
    }
  }
}

inline llvm::Instruction *unfoldConstantExpression(llvm::ConstantExpr *CE,
                                                   llvm::Instruction *InsertionPt) {

  llvm::SmallPtrSet<llvm::User*, 16> NewUsers;
  for(auto* U : CE->users()) {
    if(auto* ParentCE = llvm::dyn_cast<llvm::ConstantExpr>(U)) {
      llvm::Instruction* ParentCEReplacement = unfoldConstantExpression(ParentCE, InsertionPt);
      InsertionPt = ParentCEReplacement;
      NewUsers.insert(ParentCEReplacement);
    }
  }

  llvm::Instruction* NewI = CE->getAsInstruction(InsertionPt);
  CE->replaceUsesWithIf(NewI, [&](llvm::Use& U){
    return NewUsers.find(U.getUser()) != NewUsers.end();
  });
  CE->replaceUsesWithIf(NewI, [&](llvm::Use& U){
    if(auto* I = llvm::dyn_cast<llvm::Instruction>(U.getUser())) {
      if(I->getParent() && I->getParent()->getParent()) {
        return I->getParent()->getParent() == InsertionPt->getParent()->getParent();
      }
    }
    return false;
  });
  return NewI;
}

inline void unfoldConstantExpression(llvm::ConstantExpr *CE,
                                     const llvm::SmallVector<llvm::Instruction *> &InsertionPts) {

  for (auto *I : InsertionPts) {
    auto *NewI = unfoldConstantExpression(CE, I);
  }
}
}

// Unfold constant expression. This only works if the CE is actually used
// by an instruction.
inline void unfoldConstantExpression(llvm::ConstantExpr *CE) {
  llvm::SmallPtrSet<llvm::Function*, 16> FunctionUsers;
  constexpr_unfolding::collectFunctionUsers(CE, FunctionUsers);

  llvm::SmallVector<llvm::Instruction*> InsertionPts;
  for(auto* F : FunctionUsers) {
    if(!F->isDeclaration()) {
      InsertionPts.push_back(&(*F->getEntryBlock().getFirstInsertionPt()));
    }
  }

  if(InsertionPts.size() > 0)
    constexpr_unfolding::unfoldConstantExpression(CE, InsertionPts);
}

}
}
}

#endif
