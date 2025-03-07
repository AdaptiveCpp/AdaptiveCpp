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


#ifndef ACPP_SSCP_PCUDA_LOCAL_MEMORY_AS_MIGRATION_PASS_HPP
#define ACPP_SSCP_PCUDA_LOCAL_MEMORY_AS_MIGRATION_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// Handles statically sized local memory allocations using the
// acpp_local_memory LLVM IR annotation.
//
// This pass does two things:
//
// 1. Migrates global variables with acpp_local_memory annotation to
// to the specified local address space, inserting addressspace casts
// to generic AS where necessary.
//
// 2. Replaces allocas with acpp_local_memory annotation with global variables
// that reside in the specified local address space.
class StaticLocalMemoryPass : public llvm::PassInfoMixin<StaticLocalMemoryPass> {
public:
  StaticLocalMemoryPass(unsigned LocalMemAddressSpace);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
  unsigned LocalMemAS;
};
} // namespace compiler
}


#endif
