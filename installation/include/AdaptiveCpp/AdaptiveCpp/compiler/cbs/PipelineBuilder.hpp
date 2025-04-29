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
#ifndef HIPSYCL_PIPELINEBUILDER_HPP
#define HIPSYCL_PIPELINEBUILDER_HPP

#include <llvm/Passes/PassBuilder.h>

namespace llvm {
namespace legacy {
class PassManagerBase;
} // namespace legacy
} // namespace llvm

namespace hipsycl::compiler {

using OptLevel = llvm::OptimizationLevel;

// build the CBS pipeline for the legacy PM
void registerCBSPipelineLegacy(llvm::legacy::PassManagerBase &PM);

// build the CBS pipeline for the new PM
void registerCBSPipeline(llvm::ModulePassManager &MPM, OptLevel Opt, bool IsSscp);
} // namespace hipsycl::compiler
#endif // HIPSYCL_PIPELINEBUILDER_HPP
