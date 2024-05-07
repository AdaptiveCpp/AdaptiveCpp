/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
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

#include "hipSYCL/compiler/cbs/KernelFlattening.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>

namespace {
using namespace hipsycl::compiler;
bool inlineCallsInBasicBlock(llvm::BasicBlock &BB) {
  bool LastChanged = false, Changed = false;

  do {
    LastChanged = false;
    for (auto &I : BB) {
      if (auto *CallI = llvm::dyn_cast<llvm::CallBase>(&I)) {
        if (CallI->getCalledFunction()) {
          LastChanged = hipsycl::compiler::utils::checkedInlineFunction(CallI, "[KernelFlattening]",
                                                                        HIPSYCL_DEBUG_LEVEL_INFO);
          if (LastChanged)
            break;
        }
      }
    }
    Changed |= LastChanged;
  } while (LastChanged);

  return Changed;
}

//! \pre all contained functions are non recursive!
// todo: have a recursive-ness termination
bool inlineCallsInFunction(llvm::Function &F) {
  bool Changed = false;
  bool LastChanged;

  do {
    LastChanged = false;
    for (auto &BB : F) {
      LastChanged = inlineCallsInBasicBlock(BB);
      Changed |= LastChanged;
      if (LastChanged)
        break;
    }
  } while (LastChanged);

  return Changed;
}
} // namespace

void hipsycl::compiler::KernelFlatteningPassLegacy::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool hipsycl::compiler::KernelFlatteningPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F))
    return false;

  return inlineCallsInFunction(F);
}

llvm::PreservedAnalyses
hipsycl::compiler::KernelFlatteningPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  const auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAMProxy.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA) {
    llvm::errs() << "SplitterAnnotationAnalysis not cached.\n";
    return llvm::PreservedAnalyses::all();
  }
  if (!SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  if (!inlineCallsInFunction(F))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}

char hipsycl::compiler::KernelFlatteningPassLegacy::ID = 0;
