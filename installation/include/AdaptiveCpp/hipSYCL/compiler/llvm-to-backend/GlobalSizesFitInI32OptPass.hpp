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
#ifndef HIPSYCL_SSCP_GLOBAL_SIZES_FIT_IN_I32_OPT_HPP
#define HIPSYCL_SSCP_GLOBAL_SIZES_FIT_IN_I32_OPT_HPP

#include <llvm/IR/PassManager.h>


namespace hipsycl {
namespace compiler {

class GlobalSizesFitInI32OptPass : public llvm::PassInfoMixin<GlobalSizesFitInI32OptPass> {
public:
  GlobalSizesFitInI32OptPass(bool GlobalSizesFitInInt, int KnownGroupSizeX = -1,
                             int KnownGroupSizeY = -1, int KnownGroupSizeZ = -1);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
private:
  int KnownGroupSizeX;
  int KnownGroupSizeY;
  int KnownGroupSizeZ;
  bool GlobalSizesFitInInt;
};


// inserts llvm.assume calls to assert that x >= RangeMin && x < RangeMax.
bool insertRangeAssumptionForBuiltinCalls(llvm::Module &M, llvm::StringRef BuiltinName,
                                          long long RangeMin, long long RangeMax, bool MaxIsLessThanEqual = false);

}
}

#endif

