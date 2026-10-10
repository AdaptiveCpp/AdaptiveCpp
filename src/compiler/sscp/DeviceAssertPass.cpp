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
#include "hipSYCL/compiler/sscp/DeviceAssertPass.hpp"

namespace hipsycl {
namespace compiler {

namespace {

void replaceAssertFunction(llvm::Module &M, llvm::StringRef FunctionName,
                           llvm::Function *Replacement) {
  if(auto* F = M.getFunction(FunctionName)) {
    F->replaceAllUsesWith(Replacement);
    F->dropAllReferences();
    F->eraseFromParent();
  }
}

} // namespace

llvm::PreservedAnalyses DeviceAssertPass::run(llvm::Module &M,
                                            llvm::ModuleAnalysisManager &MAM) {

  static const char* OriginalAssertFail = "__assert_fail";
  static const char* OriginalGlibcxxAssertFail = "_ZSt21__glibcxx_assert_failPKciS0_S0_";
  replaceAssertFunction(M, OriginalAssertFail, DeviceAssertPass::getAssertFailBuiltin(M));
  replaceAssertFunction(M, OriginalGlibcxxAssertFail, DeviceAssertPass::getGlibcxxAssertFailBuiltin(M));

  return llvm::PreservedAnalyses::none();
}
}
}

