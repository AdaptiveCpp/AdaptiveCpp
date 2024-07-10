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
#ifndef HIPSYCL_SSCP_GLOBAL_INLINING_ATTRIBUTOR_PASS_HPP
#define HIPSYCL_SSCP_GLOBAL_INLINING_ATTRIBUTOR_PASS_HPP

#include <llvm/IR/PassManager.h>
#include <vector>
#include <string>

namespace hipsycl {
namespace compiler {

class GlobalInliningAttributorPass : public llvm::PassInfoMixin<GlobalInliningAttributorPass> {
public:
  GlobalInliningAttributorPass(const std::vector<std::string>& KernelName);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  std::vector<std::string> KernelNames;
};

}
}

#endif

