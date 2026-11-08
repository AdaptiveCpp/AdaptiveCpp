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
#ifndef HIPSYCL_STD_BUILTIN_TARGET_SPECIFIC_IR_MAPPER
#define HIPSYCL_STD_BUILTIN_TARGET_SPECIFIC_IR_MAPPER

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

class TargetSpecificIRMapper : public llvm::PassInfoMixin<TargetSpecificIRMapper> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

};

}
}


#endif

