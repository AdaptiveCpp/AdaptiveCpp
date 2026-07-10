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
#pragma once

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

/// After O3 instcombine, GEP expressions of the form:
///
///   %inner = getelementptr T,  ptr %base, i64 %n
///   %outer = getelementptr i8, ptr %inner, i64 C   ; C = constant byte offset
///
/// are folded back into a single typed GEP:
///
///   %combined = getelementptr T, ptr %base, i64 (%n + C/sizeof(T))
///
/// This is required because clspv --physical-storage-buffers cannot correctly
/// lower byte-granularity GEPs with negative offsets (clspv issue #1292):
/// it infers 'i8' as the element type and emits 4 separate byte loads instead
/// of a single word load, producing wrong results or a device crash.
///
/// The fold is only applied when C is an exact multiple of sizeof(T), ensuring
/// the result is representable as a typed index with no loss of precision.
class FoldChainedGEPsPass : public llvm::PassInfoMixin<FoldChainedGEPsPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

} // namespace compiler
} // namespace hipsycl
