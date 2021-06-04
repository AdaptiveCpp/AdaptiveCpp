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

#include "hipSYCL/compiler/IRUtils.hpp"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace hipsycl::compiler::utils {
llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F) {
  DT.reset();
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);
  return LI.getLoopFor(B);
}

bool checkedInlineFunction(llvm::CallBase *CI) {
  if (CI->getCalledFunction()->isIntrinsic())
    return false;

  // needed to be valid for success log
  const auto CalleeName = CI->getCalledFunction()->getName().str();

  llvm::InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR <= 10
  llvm::InlineResult ILR = llvm::InlineFunction(CI, IFI, nullptr);
  if (!static_cast<bool>(ILR)) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Failed to inline function <" << calleeName << ">: '" << ILR.message
                 << "'\n";
#else
  llvm::InlineResult ILR = llvm::InlineFunction(*CI, IFI, nullptr);
  if (!ILR.isSuccess()) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Failed to inline function <" << CalleeName << ">: '"
                 << ILR.getFailureReason() << "'\n";
#endif
    return false;
  }

  HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "LoopSplitter inlined function <"
                                          << CalleeName << ">\n";)
  return true;
}
} // namespace hipsycl::compiler::utils