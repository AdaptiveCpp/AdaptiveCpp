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
#ifndef HIPSYCL_SSCP_POINTER_TRANSLATION_ANNOTATION_PASS_HPP
#define HIPSYCL_SSCP_POINTER_TRANSLATION_ANNOTATION_PASS_HPP

#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

static constexpr const char *MetalPtrLoadNeedsTranslationMD = "acpp.metal.ptr.load.needs_translation";
static constexpr const char *MetalPtrStoreNeedsTranslationMD = "acpp.metal.ptr.store.needs_translation";

class PointerTranslationAnnotationPass
  : public llvm::PassInfoMixin<PointerTranslationAnnotationPass> {
public:
  explicit PointerTranslationAnnotationPass(unsigned GlobalAS);

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

private:
  unsigned GlobalAS;
};

} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SSCP_POINTER_TRANSLATION_ANNOTATION_PASS_HPP
