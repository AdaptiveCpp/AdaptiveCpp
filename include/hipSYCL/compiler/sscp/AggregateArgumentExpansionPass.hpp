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
#ifndef HIPSYCL_AGGREGATE_ARGUMENT_EXPANSION_PASS_HPP
#define HIPSYCL_AGGREGATE_ARGUMENT_EXPANSION_PASS_HPP

#include <llvm/IR/PassManager.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace hipsycl {
namespace compiler {


struct OriginalParamInfo {
  OriginalParamInfo(std::size_t Offset, std::size_t OriginalIndex,
                    const llvm::SmallVector<std::string> &Annotations)
      : OffsetInOriginalParam{Offset}, OriginalParamIndex{OriginalIndex}, Annotations{Annotations} {
  }

  std::size_t OffsetInOriginalParam;
  std::size_t OriginalParamIndex;
  llvm::SmallVector<std::string> Annotations;
};
// Expands aggregates into primitive function arguments. Aggregate types to expand are
// expected to be marked using the ByVal attribute.
class AggregateArgumentExpansionPass : public llvm::PassInfoMixin<AggregateArgumentExpansionPass> {
public:
  AggregateArgumentExpansionPass(const std::vector<std::string>& FunctionNames);
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
  // Returns offsets of expanded arg in the original aggregate
  const std::vector<OriginalParamInfo>* getInfosOnOriginalParams(const std::string& FunctionName) const;
private:
  std::vector<std::string> AffectedFunctionNames;
  std::unordered_map<std::string, std::vector<OriginalParamInfo>> OriginalParamInfos;
};

}
}


#endif
