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
#include "hipSYCL/compiler/llvm-to-backend/clspv/ICMPNullFixupPass.hpp"

#include <llvm/IR/Instructions.h>

namespace hipsycl {
namespace compiler {

// Pass for fixing up icmp instructions comparing a pointer with nonzero address
// space in the first operand against a nullptr with no address space.
llvm::PreservedAnalyses
ICMPNullFixupPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &) {
  bool DidTransform = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto ICmp = llvm::dyn_cast<llvm::ICmpInst>(&I)) {
        auto Op0 = ICmp->getOperand(0);
        auto Op1 = ICmp->getOperand(1);
        auto Op0PtrType = llvm::dyn_cast<llvm::PointerType>(Op0->getType());
        auto Op1PtrType = llvm::dyn_cast<llvm::PointerType>(Op1->getType());
        if (Op0PtrType && Op1PtrType) {
          if (Op0PtrType->getAddressSpace() != 0 &&
              0 == Op1PtrType->getAddressSpace()) {
            if (auto Op1Const = llvm::dyn_cast<llvm::Constant>(Op1);
                Op1Const && Op1Const->isNullValue()) {
              auto newNull = llvm::Constant::getNullValue(Op0PtrType);
              ICmp->setOperand(1, newNull);
              DidTransform = true;
            }
          }
        }
      }
    }
  }

  return DidTransform ? llvm::PreservedAnalyses::none()
                      : llvm::PreservedAnalyses::all();
}
} // namespace compiler
} // namespace hipsycl
