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
#ifndef ACPP_INDIRECT_ACCESS_UTILS_HPP
#define ACPP_INDIRECT_ACCESS_UTILS_HPP

#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Type.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

namespace hipsycl {
namespace compiler {
namespace utils {

// Note: This function should be run after optimizations and inlining!
// It does *not* handle function calls!
// TODO: Generalize this if we ever want to support partially inlined kernels
inline bool IsFunctionFreeOfIndirectAccess(llvm::Function* F) {
  if(!F)
    return true;
  for(auto& BB : *F) {
    for(auto& I : BB) {
      if(auto* ITP = llvm::dyn_cast<llvm::IntToPtrInst>(&I))
        return false;
      else if(auto* BI = llvm::dyn_cast<llvm::BitCastInst>(&I)) {
        if(BI->getType()->isPointerTy() && !BI->getSrcTy()->isPointerTy()) {
          return false;
        }
      }
      else if(auto* LI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        if(LI->getType()->isPointerTy())
          return false;
      } else if(auto* AI = llvm::dyn_cast<llvm::AtomicRMWInst>(&I)) {
        if(AI->getType()->isPointerTy())
          return false;
      } else if(auto* CB = llvm::dyn_cast<llvm::CallBase>(&I)) {
        // User functions should be inlined at the point this
        // analysis is run. So this is mainly to handle 
        // LLVM, AdaptiveCpp, backend builtins.
        if(CB->getType()->isPointerTy())
          return false;
        if(CB->isIndirectCall())
          return false;
      }
    }
  }
  return true;
}

}
}
}

#endif
