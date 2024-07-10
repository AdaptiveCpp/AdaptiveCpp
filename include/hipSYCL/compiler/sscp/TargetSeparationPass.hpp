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
#ifndef HIPSYCL_TARGET_SEPARATION_PASS_HPP
#define HIPSYCL_TARGET_SEPARATION_PASS_HPP

#include <llvm/IR/PassManager.h>
#include <string>
#include <vector>

namespace hipsycl {
namespace compiler {

class TargetSeparationPass : public llvm::PassInfoMixin<TargetSeparationPass> {
public:
  TargetSeparationPass(const std::string& KernelCompilationOptions);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  std::vector<std::string> CompilationFlags;
  std::vector<std::pair<std::string, std::string>> CompilationOptions;
};

}
}


#endif

