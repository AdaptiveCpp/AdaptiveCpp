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
#ifndef HIPSYCL_LLVM_TO_METAL_HPP
#define HIPSYCL_LLVM_TO_METAL_HPP

#include "../LLVMToBackend.hpp"

#include <optional>
#include <vector>
#include <string>
#include <unordered_set>

namespace hipsycl {
namespace compiler {

class LLVMToMetalTranslator : public LLVMToBackendTranslator{
public:
  LLVMToMetalTranslator(const std::vector<std::string>& KernelNames);
  virtual ~LLVMToMetalTranslator();

protected:
  virtual AddressSpaceMap getAddressSpaceMap() const override;
  virtual bool isKernelAfterFlavoring(llvm::Function& F) override;

  // If backend needs to set IR constants, it should do so here.
  virtual bool prepareBackendFlavor(llvm::Module& M) override;
  // Transform LLVM IR as much as required to backend-specific flavor
  virtual bool toBackendFlavor(llvm::Module &M, PassHandler& PH) override;
  virtual bool translateToBackendFormat(llvm::Module& FlavoredModule, std::string& out) override;

  // Transfers kernel properties (e.g. kernel call conventions, additional metadata) from one kernel
  // "From" to another "To". This is useful e.g. for dead argument elimination, where a new
  // kernel entrypoint with different signature will be created post optimizations.
  // This assumes that To has been created with a matching function signature from From,
  // including function and parameter attributes.
  virtual void migrateKernelProperties(llvm::Function* From, llvm::Function* To) override;
  virtual bool applyBuildOption(const std::string &Option, const std::string &Value) override;

private:
  std::vector<std::string> KernelNames;
  std::unordered_set<std::string> ActualKernelNames;
  std::optional<int> MaxArgsForFlatMode;
};

} // namespace compiler
} // namespace hipsycl

#endif
