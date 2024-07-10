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
#ifndef HIPSYCL_REFLECTION_INTROSPECT_STRUCT_PASS_HPP
#define HIPSYCL_REFLECTION_INTROSPECT_STRUCT_PASS_HPP

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <vector>

namespace hipsycl {
namespace compiler {

class IntrospectStructPass : public llvm::PassInfoMixin<IntrospectStructPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

}
}

#endif