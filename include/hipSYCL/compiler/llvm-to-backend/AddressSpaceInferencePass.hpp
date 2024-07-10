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
#ifndef HIPSYCL_SSCP_ADDRESS_SPACE_INFERENCE_PASS_HPP
#define HIPSYCL_SSCP_ADDRESS_SPACE_INFERENCE_PASS_HPP

#include <llvm/IR/PassManager.h>
#include "Utils.hpp"
#include "AddressSpaceMap.hpp"

namespace hipsycl {
namespace compiler {

class AddressSpaceInferencePass : public llvm::PassInfoMixin<AddressSpaceInferencePass> {
public:
  AddressSpaceInferencePass(const AddressSpaceMap& Map);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  AddressSpaceMap ASMap;
};

}
}

#endif

