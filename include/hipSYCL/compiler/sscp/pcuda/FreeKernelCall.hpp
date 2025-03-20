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


#ifndef ACPP_SSCP_FREE_KERNEL_CALL_PASS_HPP
#define ACPP_SSCP_FREE_KERNEL_CALL_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// Transforms code of the form:
//
// [[clang::annotate("acpp_free_kernel")]] void my_kernel(...) {...}
// ...
// my_kernel(args...);
//
// to
//
// if(__acpp_sscp_is_device)
//   my_kernel(args...)
// else {
//   const char* kernel_name = /*mangled kernel symbol name*/;
//   void** args = /*argument buffer*/
//   __pcudaKernelCall(kernel_name, args);
// }

class FreeKernelCallPass : public llvm::PassInfoMixin<FreeKernelCallPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};
}
}


#endif
