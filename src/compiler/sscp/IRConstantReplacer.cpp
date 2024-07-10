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
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include <climits>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <type_traits>

namespace hipsycl {
namespace compiler {

S1IRConstantReplacer::S1IRConstantReplacer(
    const std::unordered_map<std::string, int> &IntConstants,
    const std::unordered_map<std::string, uint64_t> &UInt64Constants,
    const std::unordered_map<std::string, std::string> &StringConstants)
    : IntConstants{IntConstants}, UInt64Constants{UInt64Constants}, StringConstants{
                                                                        StringConstants} {}

llvm::PreservedAnalyses S1IRConstantReplacer::run(llvm::Module &M,
                                                  llvm::ModuleAnalysisManager &MAM) {
  auto setConstants = [&](const auto& ConstantReplacementTable) {
    for(const auto& IC : ConstantReplacementTable) {
      if(llvm::GlobalVariable* G = M.getGlobalVariable(IC.first, true)) {
        IRConstant C{M, *G};
        C.set(IC.second);
      }
    }
  };

  setConstants(IntConstants);
  setConstants(UInt64Constants);
  setConstants(StringConstants);


  // TODO Make this more specific
  return llvm::PreservedAnalyses::none();
}
}
}
