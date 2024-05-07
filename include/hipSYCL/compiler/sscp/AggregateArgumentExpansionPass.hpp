/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
