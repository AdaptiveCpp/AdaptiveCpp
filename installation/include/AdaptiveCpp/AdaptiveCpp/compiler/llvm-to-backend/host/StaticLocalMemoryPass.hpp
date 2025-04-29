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
#ifndef ACPP_HOST_STATIC_LOCAL_MEMORY_HPP
#define ACPP_HOST_STATIC_LOCAL_MEMORY_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// On host, we need to remap globals in address space 3 to host local memory builtins.
// This is handled by this pass.
class HostStaticLocalMemoryPass : public llvm::PassInfoMixin<HostStaticLocalMemoryPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace compiler
} // namespace hipsycl

#endif
