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
#ifndef _HL_EXTRACTION_PASS_HPP_
#define _HL_EXTRACTION_PASS_HPP_

#include "llvm/IR/PassManager.h"

#include "HLTree.hpp"

namespace hipsycl {
namespace compiler {

namespace hl {

struct HLExtractionPass : llvm::PassInfoMixin<HLExtractionPass> {
  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager& FAM);

  NodePtr tree;
};

} // namespace hl

} // namespace compiler
} // namespace hipsycl

#endif